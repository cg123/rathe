from dataclasses import dataclass, field
from typing import Dict, List
from abc import ABC

from .prompt import CompletionPrompt, InstructPrompt, ChatPrompt, MessageSender, Prompt


@dataclass
class FormattedChunk:
    """A formatted piece of text.

    :cvar text: The actual text content.
    :cvar is_input: If True, this chunk is an input to a model.
    """

    text: str
    is_input: bool


def coalesce(chunks: List[FormattedChunk]) -> List[FormattedChunk]:
    """Merge consecutive chunks with the same 'is_input' status."""

    result = []
    is_input = True
    chunklets = []

    def emit():
        if chunklets:
            result.append(FormattedChunk("".join(chunklets), is_input))
            chunklets.clear()

    for chunk in chunks:
        if is_input != chunk.is_input:
            emit()
            is_input = chunk.is_input
        chunklets.append(chunk.text)
    emit()

    return result


@dataclass
class TokenizationOptions:
    """Determines specific behavior when tokenizing formatted prompts.

    :cvar generate_labels: If True, generate labels for each token indicating if
        the token is from an input chunk, defaults to True.
    :cvar input_token_id: The integer to use as the token label for input chunks,
        defaults to -100.
    :cvar add_bos: If True, a BOS (Beginning of String) token is added at the start
        of the first chunk, defaults to True.
    :cvar eos_after_output: If True, an EOS (End of String) token is added after
        each output chunk in the labels, but not the input, used as a learning
        signal for the model, defaults to False.
    """

    generate_labels: bool = True
    input_token_id: int = -100
    add_bos: bool = True
    eos_after_output: bool = False


@dataclass
class FormatResult:
    """Represents the result of a prompt formatting operation."""

    chunks: List[FormattedChunk] = field(default_factory=list)

    def add(self, text: str, is_input: bool = False):
        """Add a new chunk to the formatting result.

        :param text: The text to be added.
        :param is_input: Whether the text is an input. Defaults to False.
        """
        self.chunks.append(FormattedChunk(text, is_input))

    def coalesced(self) -> "FormatResult":
        """Returns an equivalent FormatResult, with consecutive chunks of same input
        status merged."""
        return FormatResult(coalesce(self.chunks))

    def strip_labels(self) -> "FormatResult":
        """Returns this result as a single chunk, with no input labeling."""
        return FormatResult(
            [FormattedChunk("".join([c.text for c in self.chunks]), is_input=False)]
        )

    def to_string(self):
        """Return the raw text of the formatted result."""
        return "".join([c.text for c in self.chunks])

    def to_tokens(
        self,
        tokenizer: "transformers.PreTrainedTokenizerBase",
        options: TokenizationOptions = TokenizationOptions(),
    ) -> Dict[str, "torch.Tensor"]:
        """Tokenize the chunks of text using the provided tokenizer and return a
        dictionary with the tokenized input, attention mask, and labels (if requested).

        Note: Concatenating token sequences and then decoding them can produce slightly
        different output than directly tokenizing the concatenated strings. This
        difference arises due to the way tokenizers handle the boundaries between
        different pieces of text. In the context of this function, the expected use case
        for multi-chunk results is training a language model on the prompt, with all
        input sections masked out to optimize only for the output sections. In that case
        it makes sense to tokenize the components separately and stitch them together.
        However, if you plan to use the formatted result for inference, it may be more
        appropriate to first concatenate the strings and then tokenize the resulting
        string to maintain consistency. For this, consider using the `strip_labels`
        method to return the formatted result as a single chunk, and then tokenize that
        chunk.

        :param tokenizer: An instance of transformers.PreTrainedTokenizerBase used to
            tokenize text.
        :return: A dictionary containing 'input_ids' and 'attention_mask' for each
                token, and 'labels' for each token if generate_labels is True.
        """
        import torch

        res = {
            "input_ids": [],
            "attention_mask": [],
        }

        if options.generate_labels:
            res["labels"] = []

        last_was_output = False
        for i, chunk in enumerate(self.chunks):
            tokenized = tokenizer(
                chunk.text,
                return_tensors="pt",
                add_special_tokens=True,
            )
            # remove extraneous batch dimension
            for key in tokenized:
                tokenized[key] = tokenized[key].squeeze(0)

            if tokenized["input_ids"][0] == tokenizer.bos_token_id and (
                i > 0 or not options.add_bos
            ):
                # strip BOS token from all but the first chunk
                tokenized["input_ids"] = tokenized["input_ids"][1:]
                tokenized["attention_mask"] = tokenized["attention_mask"][1:]

            if options.generate_labels:
                labels = tokenized["input_ids"]
                if chunk.is_input:
                    labels = torch.ones_like(labels)
                    labels *= options.input_token_id
                    if (
                        options.eos_after_output
                        and last_was_output
                        and res["input_ids"][-1][-1] != tokenizer.eos_token_id
                    ):
                        labels[0] = tokenizer.eos_token_id

                res["labels"].append(labels)

            for key in ("input_ids", "attention_mask"):
                res[key].append(tokenized[key])

            last_was_output = not chunk.is_input

        for key in res:
            res[key] = torch.cat(res[key], dim=-1)

        return res


class AbstractPromptFormatter(ABC):
    """An abstract base class for formatting prompts in various formats.

    This base class should be inherited by any class meant to format `Prompt` objects
    into the specific format expected by a language model.
    """

    def format(self, prompt: Prompt, special_tokens: Dict[str, str]) -> FormatResult:
        """Format a prompt into string chunks, labeled as to input status.

        :param prompt: The prompt to format.
        :param special_tokens: A dictionary mapping special token names to their string
                representations.
        """
        raise NotImplementedError()


@dataclass
class WrapperStrings:
    """Represents an optional prefix and suffix to be applied."""

    prefix: str = ""
    suffix: str = ""

    def wrap(self, text: str) -> str:
        return f"{self.prefix}{text}{self.suffix}"

    def wrap_format(self, text: str, **kwargs) -> str:
        prefix = self.prefix.format(**kwargs)
        if text is None:
            return prefix

        suffix = self.suffix.format(**kwargs)
        return f"{prefix}{text}{suffix}"


@dataclass
class AlpacaPromptFormatter(AbstractPromptFormatter):
    system_prompt: str = (
        "Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes the "
        "request.\n\n"
    )
    system_no_input_prompt: str = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
    )

    instruction_wrap = WrapperStrings("### Instruction:\n", "\n\n")
    input_wrap = WrapperStrings("### Input:\n", "\n\n")
    output_wrap = WrapperStrings("### Response:\n")

    def format(self, prompt: Prompt, special_tokens: Dict[str, str]) -> FormatResult:
        res = FormatResult()

        if isinstance(prompt, ChatPrompt) or (
            isinstance(prompt, InstructPrompt) and prompt.input
        ):
            res.add(self.system_prompt, is_input=True)
        else:
            res.add(self.system_no_input_prompt, is_input=True)

        if isinstance(prompt, InstructPrompt):
            res.add(
                self.instruction_wrap.wrap_format(prompt.instruction, **special_tokens),
                is_input=True,
            )
            if prompt.input:
                res.add(
                    self.input_wrap.wrap_format(prompt.input, **special_tokens),
                    is_input=True,
                )

            res.add(self.output_wrap.wrap_format(prompt.output, **special_tokens))
        elif isinstance(prompt, ChatPrompt):
            for message in prompt.messages:
                from_model = False
                if message.sender == MessageSender.model:
                    sender_tag = "ASSISTANT: "
                    from_model = True
                elif message.sender == MessageSender.human:
                    sender_tag = "USER: "
                else:
                    # skip system messages, i guess??
                    continue

                res.add(sender_tag, is_input=not from_model)
                if message.text is None:
                    # spot for generated answer - return immediately
                    return res.coalesced()

                res.add(f"{message.text}\n", is_input=not from_model)
        elif isinstance(prompt, CompletionPrompt):
            res.add(prompt)
        else:
            raise RuntimeError("Unsupported prompt for AlpacaPromptFormatter", prompt)

        return res.coalesced()

    @classmethod
    def wizardlm_7b(cls) -> "AlpacaPromptFormatter":
        return cls(
            system_prompt="",
            system_no_input_prompt="",
            input_wrap=WrapperStrings("\n", "\n\n"),
        )


@dataclass
class ChatPromptFormatter(AbstractPromptFormatter):
    """A generic formatter for chat prompts.

    Applies a string prefix and suffix to messages, based on the sender. Also allows for
    a system prompt prefix and a final suffix.
    """

    system_prompt: str = ""
    user_wrapper: WrapperStrings = WrapperStrings()
    model_wrapper: WrapperStrings = WrapperStrings()
    system_wrapper: WrapperStrings = WrapperStrings()
    suffix: str = ""

    def format(self, prompt: Prompt, special_tokens: Dict[str, str]) -> FormatResult:
        res = FormatResult()
        if isinstance(prompt, CompletionPrompt):
            res.add(prompt)
            return res
        else:
            prompt = prompt.as_chat()

        res.add(self.system_prompt.format(**special_tokens), is_input=True)
        for message in prompt.messages:
            wrapper = {
                MessageSender.human: self.user_wrapper,
                MessageSender.model: self.model_wrapper,
                MessageSender.system: self.system_wrapper,
            }[message.sender]

            is_input = message.sender != MessageSender.model

            prefix = wrapper.prefix.format(**special_tokens)
            res.add(prefix, is_input)

            if message.text is None:
                # spot for generated response - return immediately
                return res.coalesced()

            suffix = wrapper.suffix.format(**special_tokens)
            res.add(
                f"{message.text}{suffix}",
                is_input,
            )

        res.add(self.suffix.format(**special_tokens))
        return res.coalesced()

    @classmethod
    def oasst(cls):
        return cls(
            user_wrapper=WrapperStrings("<|prompter|>", "{eos_token}"),
            model_wrapper=WrapperStrings("<|assistant|>", "{eos_token}"),
            system_wrapper=WrapperStrings("<|system|>", "{eos_token}"),
        )

    @classmethod
    def vicuna(cls, version="1.1"):
        if version == "1.0":
            return cls(
                system_prompt="A chat between a human and an assistant.\n\n",
                user_wrapper=WrapperStrings("### Human:\n", "\n"),
                model_wrapper=WrapperStrings("### Assistant:\n", "\n"),
                system_wrapper=WrapperStrings(suffix="\n"),
                suffix="{eos_token}",
            )
        elif version == "1.1":
            return cls(
                user_wrapper=WrapperStrings("USER: ", "\n"),
                model_wrapper=WrapperStrings("ASSISTANT: ", "\n"),
                system_wrapper=WrapperStrings(suffix="\n"),
                suffix="{eos_token}",
            )
        else:
            raise RuntimeError(f"Unsupported vicuna version: {version}")

    @classmethod
    def pygmalion(
        cls,
        character: str = "Jeremy Smellston",
        persona: str = (
            "The least helpful man in the world. Arrogant, ill-informed, "
            "a little too verbose, and will actively attempt to mislead. "
            "Wears a silly little hat. Smells bad."
        ),
    ) -> "ChatPromptFormatter":
        return cls(
            system_prompt=f"{character}'s Persona: {persona}\n<START>\n",
            user_wrapper=WrapperStrings("You: ", "\n"),
            model_wrapper=WrapperStrings(f"{character}: ", "\n"),
            suffix="{eos_token}",
        )

    @classmethod
    def metharme(cls):
        return cls(
            user_wrapper=WrapperStrings("<|user|>"),
            model_wrapper=WrapperStrings("<|model|>"),
            system_wrapper=WrapperStrings("<|system|>"),
        )

    @classmethod
    def bluemoon(cls) -> "ChatPromptFormatter":
        return cls(
            system_prompt=(
                "A transcript of a roleplay between two players, LEAD and ASSOCIATE. "
                "LEAD sets up a scenario and the characters, from which ASSOCIATE then "
                "assumes a character role and continues the story for that role in "
                "response to description given by LEAD. The story and characters are "
                "developed by exchange of detailed event descriptions and character "
                "dialogs, successively given by both LEAD and ASSOCIATE.\n"
            ),
            user_wrapper=WrapperStrings("LEAD: ", "\n"),
            model_wrapper=WrapperStrings("ASSOCIATE: ", "\n"),
            suffix="{eos_token}",
        )

    @classmethod
    def chatml(cls) -> "ChatPromptFormatter":
        return cls(
            user_wrapper=WrapperStrings("<|im_start|>user\n", "<|im_end|>\n"),
            model_wrapper=WrapperStrings("<|im_start|>assistant\n", "<|im_end|>\n"),
            system_wrapper=WrapperStrings("<|im_start|>system\n", "<|im_end|>\n"),
        )


def get_formatter(name: str):
    if name == "alpaca":
        return AlpacaPromptFormatter()
    elif name == "wizardlm":
        return AlpacaPromptFormatter.wizardlm_7b()
    elif name == "vicuna":
        return ChatPromptFormatter.vicuna()
    elif name == "oasst":
        return ChatPromptFormatter.oasst()
    elif name == "bluemoon":
        return ChatPromptFormatter.bluemoon()
    elif name == "chatml":
        return ChatPromptFormatter.chatml()
    elif name == "pygmalion":
        return ChatPromptFormatter.pygmalion()
    elif name == "metharme":
        return ChatPromptFormatter.metharme()
