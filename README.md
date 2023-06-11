# Rathe

A tiny library for working with language model prompts.

Contains routines for formatting instructional and chat-based prompts into the formats expected by a variety of large language models, including but not limited to:
* Alpaca
* Open Assistant
* Vicuna
* Pygmalion

Also contains code for transforming various LLM datasets into a common representation.

## Examples

### For training
```python
from rathe import ChatPrompt, ChatMessage, MessageSender, ChatPromptFormatter, TokenizationOptions

chat = ChatPrompt(
    messages=[
        ChatMessage(MessageSender.human, "hello it is me the user, i seek a boon"),
        ChatMessage(MessageSender.model, "name your desire, fleshling"),
    ]
)

formatter = ChatPromptFormatter.vicuna()
result = formatter.format(chat, special_tokens={"eos_token": "</s>"})
tokenized = result.to_tokens(tokenizer=..., options=TokenizationOptions(generate_labels=False))
```

### For inference
```python
from rathe import AlpacaPromptFormatter, InstructPrompt

instruction = InstructPrompt(
    "In a few lines of concise proof, demonstrate that the non-trivial "
    "zeros of the Riemann zeta function have real part 1/2.",
    output=None,
)
formatter = AlpacaPromptFormatter()
result = formatter.format(instruction, special_tokens={"eos_token": "</s>"})
print(result.to_string())
```

### For use with `transformers` and `datasets` libraries
```python
import transformers
import datasets
from rathe import GenericInstructParser, ChatPromptFormatter
from rathe.pipeline import DataPipeline

parser = rathe.GenericInstructParser.dolly()
formatter = rathe.ChatPromptFormatter.vicuna()
tokenizer = transformers.LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
pipeline = DataPipeline(parser, formatter, tokenizer)

dataset = datasets.load_dataset("databricks/databricks-dolly-15k")
tokenized = dataset.map(pipeline, remove_columns=dataset['train'].column_names)
```
