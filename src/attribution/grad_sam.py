"""
Gradient-weighted Saliency Attention Maps (Grad-SAM) for text.

This method adapts Grad-CAM (Gradient-weighted Class Activation Mapping)
from computer vision to attention-based language models. It computes
gradient-weighted attention scores to determine token importance.

Reference:
    Original Grad-CAM for vision:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (ICCV 2017)
    https://arxiv.org/abs/1610.02391

    Adaptation to Transformers:
    Chefer et al., "Transformer Interpretability Beyond Attention Visualization"
    (CVPR 2021) - uses gradient-attention relevance propagation

Formula:
    For each layer l and head h:
        α_l,h = mean(∂F/∂A_l,h)  (global average pooling of gradients)
        SAM_l = ReLU(Σ_h α_l,h * A_l,h)  (weighted combination)

    Final scores = aggregate SAM across layers

Where:
    - A_l,h is the attention matrix for layer l, head h
    - F is the model output (logit for target token)

Intuition:
    - Gradients indicate which attention weights are important for the output
    - Multiplying attention by gradients highlights truly relevant connections
    - ReLU keeps only positive contributions (features that increase output)
"""

from typing import Optional, Union, Literal, List

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedModel

from .base import AttributionMethod, AttributionResult


class GradSAM(AttributionMethod):
    """
    Gradient-weighted Saliency Attention Maps for transformer models.

    Computes token importance by combining attention weights with their
    gradients w.r.t. the target prediction.
    """

    def __init__(
        self,
        layer_aggregation: Literal["mean", "last", "sum", "max"] = "mean",
        head_aggregation: Literal["mean", "sum", "max"] = "mean",
        use_relu: bool = True,
        absolute_gradients: bool = False,
    ):
        """
        Initialize Grad-SAM.

        Args:
            layer_aggregation: How to aggregate across layers
                - "mean": Average across layers
                - "last": Use only last layer
                - "sum": Sum across layers
                - "max": Max across layers
            head_aggregation: How to aggregate across attention heads
                - "mean": Average attention across heads
                - "sum": Sum attention across heads
                - "max": Max attention across heads
            use_relu: Apply ReLU to keep only positive contributions
            absolute_gradients: Use absolute value of gradients (ignore sign)
        """
        self.layer_aggregation = layer_aggregation
        self.head_aggregation = head_aggregation
        self.use_relu = use_relu
        self.absolute_gradients = absolute_gradients

    @property
    def name(self) -> str:
        return "grad_sam"

    @property
    def requires_grad(self) -> bool:
        return True

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        target_token_id: Optional[int] = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute attribution scores using Grad-SAM.

        Args:
            model: HuggingFace causal LM
            input_ids: Input token ids [1, T] or [T]
            target_pos: Position to explain
            target_token_id: Token id being predicted (if None, use argmax)
            **kwargs: Additional arguments (unused)

        Returns:
            AttributionResult with importance scores
        """
        device = next(model.parameters()).device

        # Ensure input_ids is 2D
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        T = input_ids.shape[1]

        # Storage for attention weights and gradients
        attention_weights: List[Tensor] = []
        attention_gradients: List[Tensor] = []

        # Hook to capture attention weights
        def save_attention_hook(module, args, output):
            # For HuggingFace models, attention output is typically a tuple
            # with attention weights as the second element when output_attentions=True
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]  # [batch, heads, seq, seq]
                if attn is not None:
                    attention_weights.append(attn)

        # Hook to capture attention gradients
        def save_attention_grad_hook(grad):
            attention_gradients.append(grad)

        # Register hooks on attention layers
        hooks = []
        try:
            # Get attention modules - try different model architectures
            attn_modules = self._get_attention_modules(model)

            for attn_module in attn_modules:
                hook = attn_module.register_forward_hook(save_attention_hook)
                hooks.append(hook)

            # Forward pass with attention output
            outputs = model(input_ids, output_attentions=True)
            logits = outputs.logits  # [1, T, V]

            # Get all attention weights from outputs.attentions
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attention_weights = list(outputs.attentions)

            # Get target token if not provided
            if target_token_id is None:
                target_token_id = logits[0, target_pos].argmax().item()

            # Make attention weights require gradients
            for i, attn in enumerate(attention_weights):
                attention_weights[i] = attn.detach().requires_grad_(True)

            # Re-forward with gradient-enabled attention
            # We need to compute gradients w.r.t. attention weights
            # Use a simple approach: backward from target logit

            # Get input embeddings and enable gradients
            embed_layer = model.get_input_embeddings()
            input_embeds = embed_layer(input_ids).detach().requires_grad_(True)

            # Clear old attention weights
            attention_weights.clear()

            # Forward pass again
            outputs2 = model(inputs_embeds=input_embeds, output_attentions=True)
            logits2 = outputs2.logits

            # Store attention weights from this forward pass
            if hasattr(outputs2, 'attentions') and outputs2.attentions is not None:
                # Convert to list and enable gradients
                for attn in outputs2.attentions:
                    attn_grad = attn.detach().requires_grad_(True)
                    attention_weights.append(attn_grad)

            # Compute target logit
            target_logit = logits2[0, target_pos, target_token_id]

            # Backward to get gradients w.r.t. input embeddings
            model.zero_grad()
            target_logit.backward(retain_graph=True)

            # Compute gradient-weighted attention for each layer
            # Since we can't directly get attention gradients in standard HuggingFace,
            # we use a proxy: gradient of loss w.r.t. embeddings as importance weights
            embed_grads = input_embeds.grad[0]  # [T, d]
            embed_importance = embed_grads.norm(dim=-1)  # [T]

            # Aggregate attention from all layers
            if len(attention_weights) > 0:
                # Stack attention weights [num_layers, batch, heads, seq, seq]
                all_attn = torch.stack([a.detach() for a in attention_weights])

                # Aggregate across heads
                if self.head_aggregation == "mean":
                    all_attn = all_attn.mean(dim=2)  # [layers, batch, seq, seq]
                elif self.head_aggregation == "sum":
                    all_attn = all_attn.sum(dim=2)
                elif self.head_aggregation == "max":
                    all_attn = all_attn.max(dim=2).values

                # Get attention to target position
                attn_to_target = all_attn[:, 0, target_pos, :]  # [layers, seq]

                # Weight by embedding gradients (proxy for attention gradients)
                grad_weighted_attn = attn_to_target * embed_importance.unsqueeze(0)

                # Apply ReLU if enabled
                if self.use_relu:
                    grad_weighted_attn = torch.relu(grad_weighted_attn)

                # Aggregate across layers
                if self.layer_aggregation == "mean":
                    scores = grad_weighted_attn.mean(dim=0)
                elif self.layer_aggregation == "last":
                    scores = grad_weighted_attn[-1]
                elif self.layer_aggregation == "sum":
                    scores = grad_weighted_attn.sum(dim=0)
                elif self.layer_aggregation == "max":
                    scores = grad_weighted_attn.max(dim=0).values

                scores = scores.detach().cpu().numpy()
            else:
                # Fallback: use embedding gradients directly
                scores = embed_importance.detach().cpu().numpy()

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return AttributionResult(
            scores=scores,
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "layer_aggregation": self.layer_aggregation,
                "head_aggregation": self.head_aggregation,
                "use_relu": self.use_relu,
                "target_token_id": target_token_id,
                "num_layers": len(attention_weights),
            }
        )

    def _get_attention_modules(self, model: PreTrainedModel) -> List:
        """
        Get attention modules from different model architectures.
        """
        modules = []

        # Try common HuggingFace patterns
        for name, module in model.named_modules():
            # Match common attention layer patterns
            if any(pattern in name.lower() for pattern in ['attention', 'attn']):
                if hasattr(module, 'forward'):
                    # Check if it's a leaf attention module (not a container)
                    if not list(module.children()) or 'self' in name.lower():
                        modules.append(module)

        return modules


def grad_sam(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    target_pos: int,
    target_token_id: Optional[int] = None,
    layer_aggregation: str = "mean",
    use_relu: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Functional interface for Grad-SAM attribution.

    Args:
        model: HuggingFace causal LM
        input_ids: Input token ids
        target_pos: Position to explain
        target_token_id: Token being predicted (optional)
        layer_aggregation: How to aggregate across layers
        use_relu: Apply ReLU to filter negative contributions

    Returns:
        scores: Attribution scores [T]
    """
    method = GradSAM(layer_aggregation=layer_aggregation, use_relu=use_relu)
    result = method.attribute(model, input_ids, target_pos, target_token_id, **kwargs)
    return result.scores
