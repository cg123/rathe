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
from rathe import ChatPrompt, ChatMessage, MessageSender, ChatPromptFormatter

chat = ChatPrompt(
    messages=[
        ChatMessage(MessageSender.human, "hello it is me the user, i seek a boon"),
        ChatMessage(MessageSender.model, "name your desire, fleshling"),
    ]
)

formatter = ChatPromptFormatter.vicuna()
result = formatter.format(chat, special_tokens={"eos_token": "</s>"})
tokenized = result.to_tokens(tokenizer=..., generate_labels=True, input_token_id=-100)
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
