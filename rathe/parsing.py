"""Routines for parsing common LLM dataset formats into Prompt objects."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from .prompt import ChatMessage, ChatPrompt, InstructPrompt, MessageSender, Prompt


class AbstractPromptParser(ABC):
    """An abstract base class for parsing prompts from various formats."""

    def parse(self, prompt: Dict[str, Any]) -> Prompt:
        """Parse a dictionary into a Prompt."""
        raise NotImplementedError()


class FieldMapping:
    """Class to map fields from a source dictionary to new names.

    Also optionally applies a transformation function to the retrieved values.

    :cvar name: The name(s) of the source dictionary field(s) to map.
    :cvar transformation: An optional function to transform the field value(s).
    """

    name: Union[str, List[str]]
    transformation: Optional[Callable] = None

    def __init__(
        self,
        value: "IntoFieldMapping",
        transformation: Optional[Callable] = None,
    ):
        """Initialize a FieldMapping object.

        :param value: The name(s) of the field(s) in the source data, or another
                FieldMapping to copy.
        :param transformation: An optional function to apply to the field value.
        """
        if isinstance(value, FieldMapping):
            assert transformation is None
            self.name = value.name
            self.transformation = value.transformation
        else:
            self.name = value
            self.transformation = transformation

    def get(self, object: Dict[str, Any]) -> Any:
        """Retrieve a value from the dictionary, apply a transformation function if it
        exists, and return the transformed value."""

        if isinstance(self.name, str):
            value = object.get(self.name, None)
        else:
            value = None
            for candidate in self.name:
                if candidate in object:
                    value = object[candidate]
                    break

        if self.transformation:
            value = self.transformation(value)
        return value


IntoFieldMapping = Union[str, List[str], FieldMapping]


class GenericInstructParser(AbstractPromptParser):
    """Parser for instruction datasets.

    Handles any dataset consisting of either two or three columns, with a textual
    instruction and output and optionally an additional input.
    """

    def __init__(
        self,
        instruction: IntoFieldMapping = "instruction",
        output: IntoFieldMapping = "output",
        input: Optional[IntoFieldMapping] = "input",
        instruction_prefix: Optional[str] = None,
    ):
        self.instruction = FieldMapping(instruction)
        self.output = FieldMapping(output)
        self.input = FieldMapping(input)
        self.instruction_prefix = instruction_prefix

    def parse(self, prompt: Dict[str, Any]) -> Prompt:
        instruction = self.instruction.get(prompt)
        if self.instruction_prefix:
            instruction = self.instruction_prefix + instruction

        return InstructPrompt(
            instruction,
            output=self.output.get(prompt),
            input=self.input.get(prompt) if self.input else None,
        )

    @classmethod
    def multiple_choice(
        cls, instruction_text: str = "Choose the answer that best answers the question."
    ) -> "GenericInstructParser":
        return GenericInstructParser(
            instruction="question",
            output=["solution", "explanation"],
            input=FieldMapping(
                "choices",
                lambda choices: "\n".join(f'- "{choice}"' for choice in choices),
            ),
            instruction_prefix=f"{instruction_text}\n",
        )

    @classmethod
    def jeopardy(cls) -> "GenericInstructParser":
        return GenericInstructParser(
            "question",
            FieldMapping("answer", lambda thing: f"What is {thing}"),
            "category",
            instruction_prefix=(
                "Below is a Jeopardy clue paired with input providing the "
                "category of the clue. Write a concise response that best "
                "answers the clue given the category."
            ),
        )

    @classmethod
    def alpaca(cls) -> "GenericInstructParser":
        return GenericInstructParser()

    @classmethod
    def oasst(cls) -> "GenericInstructParser":
        return GenericInstructParser("INSTRUCTION", "RESPONSE", None)

    @classmethod
    def tldr(cls) -> "GenericInstructParser":
        return GenericInstructParser("article", "summary", None)

    @classmethod
    def gpteacher(cls) -> "GenericInstructParser":
        return GenericInstructParser("instruction", "response", "input")

    @classmethod
    def dolly(cls) -> "GenericInstructParser":
        return GenericInstructParser("instruction", "response", "context")

    @classmethod
    def gpt4all(cls) -> "GenericInstructParser":
        return GenericInstructParser("prompt", "response")


@dataclass
class ShareGPTParser(AbstractPromptParser):
    """Parser for conversation-style datasets.

    :cvar key: The key in the source dictionary that contains the conversation data.
        Defaults to "conversations".
    """

    key: str = "conversations"

    def parse(self, prompt: Dict[str, Any]) -> Prompt:
        messages = []
        for msg in prompt[self.key]:
            sender = {
                "human": MessageSender.human,
                "user": MessageSender.human,
                "gpt": MessageSender.model,
                "bing": MessageSender.model,
                "chatgpt": MessageSender.model,
                "bard": MessageSender.model,
                "system": MessageSender.system,
            }[msg["from"]]
            messages.append(ChatMessage(sender, msg["value"]))
        return ChatPrompt(messages)


@dataclass
class CompletionParser(AbstractPromptParser):
    """A parser for raw text completion datasets.

    :cvar key: The key in the source dictionary that contains the raw text data.
        Defaults to "text".
    """

    key: str = "text"

    def parse(self, prompt: Dict[str, Any]) -> Prompt:
        return prompt[self.key]


def get_parser(type_: str) -> AbstractPromptParser:
    if type_ == "alpaca":
        return GenericInstructParser.alpaca()
    elif type_ == "sharegpt":
        return ShareGPTParser()
    elif type_ == "completion":
        return CompletionParser()
    elif type_ == "jeopardy":
        return GenericInstructParser.jeopardy()
    elif type_ == "oasst":
        return GenericInstructParser.oasst()
    elif type_ == "gpteacher":
        return GenericInstructParser.gpteacher()
    elif type_ == "gpt4all":
        return GenericInstructParser.gpt4all()
    elif type_ == "reflection":
        raise NotImplementedError()
    elif type_ == "explainchoice":
        return GenericInstructParser.multiple_choice(
            instruction_text="Choose the answer that best answers the question. "
            "Explain your reasoning."
        )
    elif type_ == "concisechoice":
        return GenericInstructParser.multiple_choice(
            instruction_text="Choose the answer that best answers the question. "
            "Be concise in your response."
        )
    elif type_ == "summarizetldr":
        return GenericInstructParser.tldr()
    else:
        raise RuntimeError(f"Unknown parser type: {type_}")
