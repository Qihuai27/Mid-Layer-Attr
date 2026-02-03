"""
DePass: Decomposed Forward Pass Attribution Method.

Migrated from: https://github.com/xxx/Decomposed-Forward-Pass
Paper: [NeurIPS 2025] DePass: Unified Feature Attributing by Simple Decomposed Forward Pass

DePass traces token contributions through transformer layers by decomposing
the forward pass. It initializes an attribution state tensor and propagates
it through attention and MLP layers to track how each input token contributes
to the final hidden states.

Key Algorithm:
1. Initialize attribution_state[i, i, :] = hidden_state[i, :] at layer 0
2. For each layer:
   - Decompose attention: distribute attention outputs based on attention weights
   - Decompose MLP: distribute MLP outputs based on gate projection contributions
3. Final scores are computed via lm_head projection

Supported Models:
- LLaMA family (LLaMA, LLaMA-2, LLaMA-3)
- Qwen family (Qwen, Qwen2, Qwen3)
"""

import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from transformers import PreTrainedModel

from .base import AttributionMethod, AttributionResult


def capture_sdpa():
    """Capture scaled dot product attention parameters."""
    original_sdpa = F.scaled_dot_product_attention
    masks, dropouts, causals = [], [], []

    def my_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        masks.append(attn_mask)
        dropouts.append(dropout_p)
        causals.append(is_causal)
        return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

    return my_sdpa, lambda: (masks, dropouts, causals)


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat key/value heads for grouped query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def get_rmsnorm_scaling(norm_layer, hidden_states: Tensor) -> Tensor:
    """Compute RMSNorm scaling factors."""
    target_dtype = hidden_states.dtype
    weight = norm_layer.weight.to(device=hidden_states.device, dtype=target_dtype)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    scale_factor = torch.rsqrt(variance + norm_layer.variance_epsilon)
    return weight * scale_factor.to(dtype=target_dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half for rotary embeddings."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_llama(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings (LLaMA style)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_qwen(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings (Qwen3 style)."""
    # Qwen3 cos/sin shape: [bsz, seq_len, head_dim]
    # q/k shape: [bsz, num_heads, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)  # [bsz, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_attention_weights_mha(
    state_norm: Tensor,
    attn_layer,
    layer_idx: int,
    attn_mask,
    dropout_p: float,
    is_causal: bool,
    apply_rotary_pos_emb,
    model_name: str,
    model_config=None,
    rotary_emb=None,
) -> Tensor:
    """Compute multi-head attention weights."""
    q_proj, k_proj = attn_layer.q_proj, attn_layer.k_proj
    query_states = q_proj(state_norm)
    key_states = k_proj(state_norm)

    bsz, seq_len, hidden_dim = query_states.shape

    # Get num_heads from config or attn_layer
    if hasattr(attn_layer, 'num_heads'):
        num_heads = attn_layer.num_heads
        num_kv_heads = attn_layer.num_key_value_heads
        head_dim = attn_layer.head_dim
    elif model_config is not None:
        num_heads = model_config.num_attention_heads
        num_kv_heads = model_config.num_key_value_heads
        # Qwen3 has explicit head_dim in config that differs from hidden_size // num_heads
        head_dim = getattr(model_config, 'head_dim', hidden_dim // num_heads)
    else:
        num_heads = hidden_dim // 128  # Default head_dim=128
        num_kv_heads = num_heads
        head_dim = 128

    position_ids = torch.arange(seq_len, dtype=torch.long, device=state_norm.device).unsqueeze(0)

    # Calculate head_dim from actual output shape if needed
    q_out_dim = query_states.shape[-1]
    k_out_dim = key_states.shape[-1]
    q_head_dim = q_out_dim // num_heads
    k_head_dim = k_out_dim // num_kv_heads

    query_states = query_states.view(bsz, seq_len, num_heads, q_head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, seq_len, num_kv_heads, k_head_dim).transpose(1, 2)

    # Get rotary embeddings
    rope = rotary_emb if rotary_emb is not None else getattr(attn_layer, 'rotary_emb', None)
    if rope is None:
        raise ValueError("Cannot find rotary_emb in attn_layer or rotary_emb argument")

    if model_name == "llama":
        cos, sin = rope(state_norm, position_ids)
    elif model_name == "qwen":
        # Qwen3 rotary_emb takes (x, position_ids)
        cos, sin = rope(state_norm, position_ids)
    else:
        cos, sin = rope(state_norm, position_ids)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, num_heads // num_kv_heads)

    L, S = query_states.size(-2), key_states.size(-2)
    scale_factor = 1 / math.sqrt(query_states.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query_states.dtype, device=query_states.device)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query_states.device).tril(diagonal=0)
        attn_bias.masked_fill_(~temp_mask, float("-inf"))
        attn_bias = attn_bias.to(query_states.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(~attn_mask, float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    attn_weight = query_states @ key_states.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight


class DePassManager:
    """
    DePass (Decomposed Forward Pass) Manager.

    This class manages the decomposition of forward pass to trace token contributions
    through transformer layers.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer=None,
        mlp_softmax_temp: float = 0.1,
        ignore_special_token: bool = True,
        mlp_decomposed_function: str = "softmax",
    ):
        """
        Initialize DePass manager.

        Args:
            model: HuggingFace transformer model (LLaMA or Qwen)
            tokenizer: Tokenizer (optional, for decoding)
            mlp_softmax_temp: Temperature for softmax in MLP decomposition
            ignore_special_token: Whether to ignore special tokens in attribution
            mlp_decomposed_function: MLP decomposition method ("softmax", "relu", "max", "linear", "taylor")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_heads = model.config.num_attention_heads
        self.num_key_value_heads = model.config.num_key_value_heads
        self.num_layers = model.config.num_hidden_layers
        # Qwen3 has explicit head_dim in config that differs from hidden_size // num_heads
        self.head_dim = getattr(model.config, 'head_dim', model.config.hidden_size // self.num_heads)
        self.hidden_dim = model.config.hidden_size
        self.mlp_softmax_temp = mlp_softmax_temp
        self.ignore_special_token = ignore_special_token

        # Select MLP decomposition function
        if mlp_decomposed_function == "softmax":
            self.mlp_decomposed_compute = self._mlp_decomposed_compute_softmax
        elif mlp_decomposed_function == "taylor":
            self.mlp_decomposed_compute = self._mlp_decomposed_compute_taylor
        else:
            self.mlp_decomposed_compute = self._mlp_decomposed_compute_softmax

        # Detect model type
        model_type = getattr(model.config, "model_type", "").lower()
        if "llama" in model_type:
            self.apply_rotary_pos_emb = apply_rotary_pos_emb_llama
            self.model_name = "llama"
        elif "qwen" in model_type:
            self.apply_rotary_pos_emb = apply_rotary_pos_emb_qwen
            self.model_name = "qwen"
        else:
            self.apply_rotary_pos_emb = apply_rotary_pos_emb_llama
            self.model_name = "llama"

        # State storage
        self.states = None
        self.masks = None
        self.dropouts = None
        self.causals = None
        self.attribute_len = None

    def _get_states(self, input_ids: Tensor) -> List[Tensor]:
        """Get intermediate states from forward pass."""
        original_sdpa = F.scaled_dot_product_attention
        my_sdpa, get_captured = capture_sdpa()
        F.scaled_dot_product_attention = my_sdpa

        # Hook to capture states
        block_inputs = {}
        attn_intermediates = {}
        pre_norm_hidden = {}
        hook_handles = []

        def get_block_pre_hook(layer_idx):
            def hook(module, inputs):
                block_inputs[layer_idx] = inputs[0]
            return hook

        def get_self_attn_hook(layer_idx):
            def hook(module, inputs, output):
                attn_out = output[0] if isinstance(output, tuple) else output
                device = attn_out.device
                block_input = block_inputs.get(layer_idx, torch.zeros_like(attn_out))
                if block_input.device != device:
                    block_input = block_input.to(device)
                attn_intermediates[layer_idx] = block_input + attn_out.clone()
            return hook

        for idx, block in enumerate(self.model.model.layers):
            hook_handles.append(block.register_forward_pre_hook(get_block_pre_hook(idx)))
            hook_handles.append(block.self_attn.register_forward_hook(get_self_attn_hook(idx)))

        def capture_pre_norm_hook(module, input, output):
            pre_norm_hidden['last'] = input[0].detach()

        handle = self.model.model.norm.register_forward_hook(capture_pre_norm_hook)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)

        # Cleanup hooks
        handle.remove()
        for h in hook_handles:
            h.remove()

        F.scaled_dot_product_attention = original_sdpa
        masks, dropouts, causals = get_captured()

        # If SDPA wasn't called (e.g., eager attention), create default values
        if len(masks) == 0:
            masks = [None] * self.num_layers
            dropouts = [0.0] * self.num_layers
            causals = [True] * self.num_layers

        # Build state stream
        hidden_states = outputs.hidden_states
        attn_intermediate_list = [attn_intermediates[i] for i in sorted(attn_intermediates.keys())]

        stream = []
        for i in range(len(attn_intermediate_list)):
            stream.append(hidden_states[i])
            stream.append(attn_intermediate_list[i])
        stream.append(pre_norm_hidden['last'])

        self.states = stream
        self.masks = masks
        self.dropouts = dropouts
        self.causals = causals
        self.attribute_len = stream[0].shape[1]

        return stream

    def _attn_decomposed_compute(
        self,
        layer_idx: int,
        state: Tensor,
        masks: List,
        dropouts: List,
        causals: List,
        attribute_state: Tensor,
    ) -> Tensor:
        """Decomposed attention computation."""
        layer = self.model.model.layers[layer_idx]
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        ln = layer.input_layernorm
        state_norm = ln(state)

        # Get rotary_emb from model.model level for Qwen3
        rotary_emb = getattr(self.model.model, 'rotary_emb', None)

        attn_weight = compute_attention_weights_mha(
            state_norm, layer.self_attn, layer_idx,
            masks[layer_idx], dropouts[layer_idx], causals[layer_idx],
            self.apply_rotary_pos_emb, self.model_name,
            model_config=self.model.config,
            rotary_emb=rotary_emb
        )

        W_diag_elements = get_rmsnorm_scaling(ln, state)
        attribute_state_norm = attribute_state * W_diag_elements

        target_device = attribute_state.device
        target_dtype = v_proj.weight.dtype

        v_proj_weight = v_proj.weight.view(
            self.num_key_value_heads, self.head_dim, self.hidden_dim
        ).to(device=target_device, dtype=target_dtype)
        o_proj_weight = o_proj.weight.view(
            self.hidden_dim, self.num_heads, self.head_dim
        ).to(device=target_device, dtype=target_dtype)

        attn_weight = attn_weight[0].to(device=target_device, dtype=target_dtype)
        attribute_state_norm = attribute_state_norm.to(dtype=target_dtype)

        attribute_values = torch.einsum("kqi,nhi->knqh", attribute_state_norm, v_proj_weight)
        attribute_values = repeat_kv(attribute_values, self.num_heads // self.num_key_value_heads)
        vo_attribute = torch.einsum("knqh,jnh->kqnj", attribute_values, o_proj_weight)
        attribute_state += torch.einsum("iknd, nqk -> iqd", vo_attribute, attn_weight)

        return attribute_state

    def _mlp_decomposed_compute_softmax(
        self,
        layer_idx: int,
        state: Tensor,
        attribute_state: Tensor,
        mlp_softmax_temp: float = 0.1,
        ignore_special_token: bool = True,
        cur_num: int = None,
    ) -> Tensor:
        """Decomposed MLP computation using softmax distribution."""
        target_device = attribute_state.device
        layer = self.model.model.layers[layer_idx]
        ln = layer.post_attention_layernorm
        gate_proj = layer.mlp.gate_proj
        target_dtype = gate_proj.weight.dtype

        state = state.to(device=target_device, dtype=target_dtype)
        state_norm = ln(state)
        W_diag_elements = get_rmsnorm_scaling(ln, state).to(device=target_device, dtype=target_dtype)
        attribute_state_norm = attribute_state.to(dtype=target_dtype) * W_diag_elements

        gate_ratio = gate_proj(attribute_state_norm).transpose(-2, -1)

        if ignore_special_token:
            last_minus_first = gate_ratio[-1:]
            gate_ratio = torch.cat([
                torch.zeros_like(gate_ratio[:1]),
                gate_ratio[1:-1],
                last_minus_first
            ], dim=0)

        gate_ratio[0, :, 0] = 1.0
        gate_ratio[-1, :, :] = torch.where(
            torch.abs(gate_ratio[-1, :, :]) < 1e-5,
            torch.zeros_like(gate_ratio[-1, :, :]),
            gate_ratio[-1, :, :]
        )
        zero_cols = (gate_ratio[:, :, :] == 0).all(dim=0)
        gate_ratio[-1, :, :] = torch.where(zero_cols, torch.ones_like(gate_ratio[-1, :, :]), gate_ratio[-1, :, :])

        mask = (gate_ratio != 0).float()
        gate_ratio.masked_fill_(mask == 0, float('-inf'))
        gate_ratio = torch.softmax(gate_ratio / mlp_softmax_temp, dim=0) * mask

        attribute_mlp_values = torch.zeros_like(attribute_state)
        mlp_token_compute_num = state_norm.shape[1]
        num_batches = (state_norm.shape[1] + mlp_token_compute_num - 1) // mlp_token_compute_num

        for i in range(num_batches):
            start_idx = i * mlp_token_compute_num
            end_idx = min((i + 1) * mlp_token_compute_num, state_norm.shape[1])

            mlp_gate = self._get_per_ffn2_values(state_norm[:, start_idx:end_idx, :], layer_idx).to(target_device)
            gate_slice = gate_ratio[:, :, start_idx:end_idx].to(target_device)
            mlp_gate = mlp_gate.squeeze(0)
            gate_slice_trans = gate_slice.permute(0, 2, 1)

            weight = layer.mlp.down_proj.weight.to(target_device)
            mlp_gate_expanded = mlp_gate.unsqueeze(0).expand(gate_slice_trans.size(0), -1, -1)
            mlp_gate = mlp_gate_expanded * gate_slice_trans

            k, n, i_dim = mlp_gate.shape
            d = weight.size(0)
            mlp_gate = mlp_gate.view(k * n, i_dim)
            mlp_gate = mlp_gate.to(dtype=weight.dtype)
            output = mlp_gate @ weight.T
            output = output.view(k, n, d)
            attribute_mlp_values[:, start_idx:end_idx, :] += output

        attribute_state += attribute_mlp_values
        return attribute_state

    def _mlp_decomposed_compute_taylor(
        self,
        layer_idx: int,
        state: Tensor,
        attribute_state: Tensor,
        mlp_softmax_temp: float = 0.1,
        ignore_special_token: bool = True,
        cur_num: int = None,
    ) -> Tensor:
        """Decomposed MLP computation using Taylor approximation."""
        target_device = attribute_state.device
        layer = self.model.model.layers[layer_idx]
        ln = layer.post_attention_layernorm
        gate_proj = layer.mlp.gate_proj
        target_dtype = gate_proj.weight.dtype

        state = state.to(device=target_device, dtype=target_dtype)
        state_norm = ln(state)
        W_diag_elements = get_rmsnorm_scaling(ln, state).to(device=target_device, dtype=target_dtype)
        attribute_state_norm = attribute_state.to(dtype=target_dtype) * W_diag_elements

        mlp = layer.mlp
        gate = mlp.gate_proj(state_norm)
        gate_act = mlp.act_fn(gate)
        coef = gate_act / (gate + 1e-8)

        gate_attribute = mlp.gate_proj(attribute_state_norm)
        gate_attribute = gate_attribute * coef
        attribute_mlp_values = mlp.down_proj(gate_attribute * mlp.up_proj(state_norm))

        attribute_state += attribute_mlp_values.to(attribute_state.device)
        return attribute_state

    def _get_per_ffn2_values(self, state: Tensor, layer_idx: int) -> Tensor:
        """Get per-neuron FFN values."""
        layer = self.model.model.layers[layer_idx].mlp
        gate_up_output = layer.act_fn(layer.gate_proj(state)) * layer.up_proj(state)
        return gate_up_output

    def compute_decomposed_state(
        self,
        input_ids: Tensor,
        start_layer_idx: int = 0,
        single_compute_token: int = None,
    ) -> Tensor:
        """
        Compute decomposed attribution state.

        Args:
            input_ids: Input token IDs [1, T]
            start_layer_idx: Layer to start decomposition from
            single_compute_token: Batch size for token computation

        Returns:
            attribution_state: Tensor of shape [T, T, D]
        """
        self._get_states(input_ids)

        masks = self.masks
        dropouts = self.dropouts
        causals = self.causals
        states = self.states
        attribute_len = self.attribute_len

        if single_compute_token is None:
            single_compute_token = attribute_len + 1

        forward_num = math.ceil(attribute_len / single_compute_token)
        attribute_state_list = []

        with torch.no_grad():
            for cur_num in range(forward_num):
                start_token_idx = cur_num * single_compute_token
                end_token_idx = min((cur_num + 1) * single_compute_token, attribute_len)
                cur_compute_token_len = end_token_idx - start_token_idx

                attribute_state = torch.zeros(
                    cur_compute_token_len, attribute_len, self.hidden_dim,
                    dtype=next(self.model.parameters()).dtype,
                    device=self.model.device
                )

                idx_range = torch.arange(start_token_idx, end_token_idx)
                local_idx = idx_range - start_token_idx
                attribute_state[local_idx, idx_range, :] = states[2 * start_layer_idx][0][idx_range, :]

                if self.ignore_special_token and cur_num != 0:
                    special_token_state = torch.zeros(
                        1, attribute_state.shape[1], attribute_state.shape[2],
                        dtype=next(self.model.parameters()).dtype
                    ).to(self.model.device)
                    special_token_state[0, 0, :] = states[2 * start_layer_idx][0][0, :]
                    attribute_state = torch.cat([special_token_state, attribute_state], dim=0)

                attribute_state_sum = torch.sum(attribute_state, dim=0)
                left_state = states[2 * start_layer_idx][0] - attribute_state_sum
                attribute_state = torch.cat([attribute_state, left_state.unsqueeze(0)], dim=0)

                # Propagate through layers
                for layer_idx in range(start_layer_idx, self.num_layers):
                    state = states[layer_idx * 2]
                    attribute_state = self._attn_decomposed_compute(
                        layer_idx, state, masks, dropouts, causals, attribute_state
                    )
                    state = states[layer_idx * 2 + 1]
                    attribute_state = self.mlp_decomposed_compute(
                        layer_idx, state, attribute_state, self.mlp_softmax_temp,
                        ignore_special_token=self.ignore_special_token, cur_num=cur_num
                    )

                attribute_state = attribute_state[:-1, :, :]
                if self.ignore_special_token and cur_num != 0:
                    attribute_state = attribute_state[1:, :, :]

                # Final layernorm
                state = states[-1][0].to(attribute_state.device)
                ln = self.model.model.norm
                W_diag_elements = get_rmsnorm_scaling(ln, state)
                attribute_state = attribute_state * W_diag_elements

                attribute_state_list.append(attribute_state.clone().detach().cpu())

            attribute_state = torch.cat(attribute_state_list, dim=0)
            attribute_state = attribute_state.transpose(0, 1)

        return attribute_state

    def compute_attribution_scores(
        self,
        input_ids: Tensor,
        target_pos: int,
        target_token_id: int = None,
    ) -> np.ndarray:
        """
        Compute attribution scores for each input token.

        Args:
            input_ids: Input token IDs [1, T]
            target_pos: Position to explain
            target_token_id: Token ID being predicted (if None, use argmax)

        Returns:
            scores: Attribution scores [T]
        """
        attribute_state = self.compute_decomposed_state(input_ids)

        # Get target token if not provided
        if target_token_id is None:
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, target_pos]
                target_token_id = logits.argmax().item()

        # Compute scores via lm_head
        lm_head = self.model.lm_head
        attr_at_pos = attribute_state[target_pos].to(
            device=self.model.device, dtype=lm_head.weight.dtype
        )
        attribute_logits = lm_head(attr_at_pos)
        scores = attribute_logits[:, target_token_id].float().detach().cpu().numpy()

        return scores


class DePass(AttributionMethod):
    """
    DePass Attribution Method.

    Computes token-level importance scores by decomposing the forward pass
    and tracing how each input token contributes to the final prediction.

    Supported models: LLaMA, Qwen families
    """

    def __init__(
        self,
        mlp_softmax_temp: float = 0.1,
        ignore_special_token: bool = True,
        mlp_decomposed_function: str = "softmax",
    ):
        """
        Initialize DePass attribution method.

        Args:
            mlp_softmax_temp: Temperature for MLP decomposition softmax
            ignore_special_token: Whether to ignore special tokens
            mlp_decomposed_function: MLP decomposition method ("softmax" or "taylor")
        """
        self.mlp_softmax_temp = mlp_softmax_temp
        self.ignore_special_token = ignore_special_token
        self.mlp_decomposed_function = mlp_decomposed_function
        self._manager = None

    @property
    def name(self) -> str:
        return "DePass"

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Tensor,
        target_pos: int,
        target_token_id: int = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute attribution scores using DePass.

        Args:
            model: HuggingFace model (LLaMA or Qwen)
            input_ids: Input token IDs [1, T] or [T]
            target_pos: Position to explain
            target_token_id: Token ID being predicted (optional)

        Returns:
            AttributionResult with importance scores
        """
        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Initialize manager
        self._manager = DePassManager(
            model=model,
            mlp_softmax_temp=self.mlp_softmax_temp,
            ignore_special_token=self.ignore_special_token,
            mlp_decomposed_function=self.mlp_decomposed_function,
        )

        # Compute scores
        scores = self._manager.compute_attribution_scores(
            input_ids, target_pos, target_token_id
        )

        return AttributionResult(
            scores=scores,
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "mlp_softmax_temp": self.mlp_softmax_temp,
                "mlp_decomposed_function": self.mlp_decomposed_function,
                "model_type": self._manager.model_name,
            }
        )


def depass(
    model: PreTrainedModel,
    input_ids: Tensor,
    target_pos: int,
    target_token_id: int = None,
    mlp_softmax_temp: float = 0.1,
    **kwargs,
) -> np.ndarray:
    """
    Functional interface for DePass attribution.

    Args:
        model: HuggingFace model (LLaMA or Qwen)
        input_ids: Input token IDs
        target_pos: Position to explain
        target_token_id: Token being predicted (optional)
        mlp_softmax_temp: Temperature for MLP decomposition

    Returns:
        scores: Attribution scores [T]
    """
    method = DePass(mlp_softmax_temp=mlp_softmax_temp, **kwargs)
    result = method.attribute(model, input_ids, target_pos, target_token_id)
    return result.scores
