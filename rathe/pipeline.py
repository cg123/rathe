from dataclasses import dataclass
from typing import Dict
from .parsing import AbstractPromptParser
from .formatting import AbstractPromptFormatter, TokenizationOptions
from transformers import PreTrainedTokenizerBase


@dataclass
class DataPipeline:
    """Handles the complete process of transforming and tokenizing data.

    Uses a prompt parser, formatter, and tokenizer to transform a dataset
    into tokenized examples to feed to a language model.
    """

    parser: AbstractPromptParser
    formatter: AbstractPromptFormatter
    tokenizer: PreTrainedTokenizerBase
    options: TokenizationOptions = TokenizationOptions()
    batched: bool = False

    def process_single(self, example: Dict) -> Dict:
        """Process a single example."""
        formatted = self.formatter.format(
            self.parser.parse(example), self.tokenizer.special_tokens_map
        )
        return formatted.to_tokens(self.tokenizer, options=self.options)

    def __call__(self, examples: Dict) -> Dict:
        """Process an example or set of examples.

        Designed to be used with the `datasets` library's `Dataset.map` method
        to process an entire `Dataset`.
        """
        batch_size = None
        for key in examples:
            if isinstance(examples[key], list):
                batch_size = len(examples[key])
            break

        if (not self.batched) or batch_size is None:
            return self.process_single(examples)

        res = {}
        for idx in range(batch_size):
            example = {key: examples[key][idx] for key in examples}
            processed = self.process_single(example)
            for key in processed:
                if key not in res:
                    res[key] = []
                res[key].append(processed[key])

        return res
