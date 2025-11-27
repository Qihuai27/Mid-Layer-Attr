"""
ICL prompt construction utilities.
"""

from typing import Dict, List, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer

from ..anchors.definitions import TaskConfig


class ICLWrapper:
    """
    Wrapper for constructing In-Context Learning prompts.

    Combines demonstrations and test examples into formatted prompts.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
    ):
        """
        Initialize the wrapper.

        Args:
            task_config: Task configuration with prompt template
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length (defaults to tokenizer's max)
        """
        self.task_config = task_config
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length

    def format_demonstration(self, example: Dict) -> str:
        """Format a single demonstration example."""
        text = example[self.task_config.text_field]
        label = example[self.task_config.label_field]
        return self.task_config.format_example(text, label)

    def format_test(self, example: Dict) -> str:
        """Format a test example (without label)."""
        text = example[self.task_config.text_field]
        return self.task_config.format_example(text, label=None)

    def build_prompt(
        self,
        demonstrations: List[Dict],
        test_example: Dict,
        separator: str = "\n",
    ) -> str:
        """
        Build a complete ICL prompt.

        Args:
            demonstrations: List of demonstration examples
            test_example: Test example to classify
            separator: String to join demonstrations

        Returns:
            Complete prompt string
        """
        # Format demonstrations
        demo_strs = [self.format_demonstration(d) for d in demonstrations]

        # Format test example
        test_str = self.format_test(test_example)

        # Combine
        prompt = separator.join(demo_strs + [test_str])

        return prompt

    def wrap_dataset(
        self,
        demonstrations: Dataset,
        test_dataset: Dataset,
    ) -> Dataset:
        """
        Wrap a test dataset with demonstrations.

        Args:
            demonstrations: Dataset of demonstration examples
            test_dataset: Dataset of test examples

        Returns:
            Dataset with 'sentence' field containing complete prompts
        """
        demo_list = [demonstrations[i] for i in range(len(demonstrations))]

        def wrap_example(example):
            prompt = self.build_prompt(demo_list, example)
            return {"sentence": prompt, "label": example[self.task_config.label_field]}

        wrapped = test_dataset.map(wrap_example)
        return wrapped

    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_field: str = "sentence",
    ) -> Dataset:
        """
        Tokenize a wrapped dataset.

        Args:
            dataset: Dataset with text field
            text_field: Name of the text field to tokenize

        Returns:
            Tokenized dataset
        """
        def tokenize_fn(examples):
            return self.tokenizer(
                examples[text_field],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )

        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=[text_field],
        )

        return tokenized


def wrap_icl_prompt(
    demonstrations: Dataset,
    test_example: Dict,
    task_config: TaskConfig,
) -> str:
    """
    Convenience function to wrap a single test example.

    Args:
        demonstrations: Demonstration examples
        test_example: Test example
        task_config: Task configuration

    Returns:
        Complete ICL prompt string
    """
    demo_list = [demonstrations[i] for i in range(len(demonstrations))]

    # Build prompt
    parts = []
    for demo in demo_list:
        text = demo[task_config.text_field]
        label = demo[task_config.label_field]
        parts.append(task_config.format_example(text, label))

    # Add test example
    test_text = test_example[task_config.text_field]
    parts.append(task_config.format_example(test_text, label=None))

    return "\n".join(parts)
