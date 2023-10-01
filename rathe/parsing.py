"""Routines for parsing common LLM dataset formats into Prompt objects."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Optional, Union

from .prompt import ChatMessage, ChatPrompt, InstructPrompt, MessageSender, Prompt, CompletionPrompt
from .rp import RoleplayPrompt, RoleplayCharacter


class PromptParser(ABC):
    """An abstract base class for parsing prompts from various formats."""

    @abstractmethod
    def parse(self, prompt: Dict[str, Any]) -> Prompt:
        """Parse a dictionary into a Prompt."""
        ...


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


class GenericInstructParser(PromptParser):
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
class ShareGPTParser(PromptParser):
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
class CompletionParser(PromptParser):
    """A parser for raw text completion datasets.

    :cvar key: The key in the source dictionary that contains the raw text data.
        Defaults to "text".
    """

    key: str = "text"
    prefix_key: Optional[str] = None

    def parse(self, prompt: Dict[str, Any]) -> Prompt:
        text = prompt[self.key]
        prefix = prompt[self.prefix_key] if self.prefix_key else None
        return CompletionPrompt(text, prefix=prefix)


@dataclass
class OrcaStyleParser(PromptParser):
    """Parser for OpenOrca dataset."""

    instruction_field: str = "question"
    output_field: str = "response"
    system_prompt_field: str = "system_prompt"

    def parse(self, prompt: Dict[str, Any]) -> Prompt:
        messages = [
            ChatMessage(MessageSender.human, prompt[self.instruction_field]),
            ChatMessage(MessageSender.model, prompt[self.output_field]),
        ]
        if self.system_prompt_field in prompt and prompt[self.system_prompt_field]:
            messages.insert(
                0, ChatMessage(MessageSender.system, prompt[self.system_prompt_field])
            )
        return ChatPrompt(messages)

    @classmethod
    def open_orca(cls) -> PromptParser:
        return OrcaStyleParser()

    @classmethod
    def orca_mini(cls) -> PromptParser:
        return OrcaStyleParser(
            instruction_field="instruction",
            output_field="output",
            system_prompt_field="system",
        )

    @classmethod
    def dolphin(cls) -> PromptParser:
        return OrcaStyleParser(
            instruction_field="input",
            output_field="output",
            system_prompt_field="instruction",
        )


@dataclass
class RoleplayForumParser(PromptParser):
    username_key: str = "username"
    name_key: str = "char_name"
    bio_key: str = "bio"
    output_key: str = "reply"

    history_key: str = "context"
    message_sender_key: str = "username"
    message_text_key: str = "text"

    def parse(self, prompt: Dict[str, Any]) -> Prompt:
        bot_char = RoleplayCharacter(prompt[self.name_key], prompt[self.bio_key])
        output = prompt[self.output_key]

        messages = []
        for msg in prompt[self.history_key]:
            sender_name = msg[self.message_sender_key]
            sender = (
                MessageSender.model
                if sender_name == prompt[self.username_key]
                else MessageSender.human
            )
            messages.append(ChatMessage(sender, text=msg[self.message_text_key]))
        messages.append(ChatMessage(MessageSender.model, output))

        return RoleplayPrompt(messages, bot_char)


class PippaParser(PromptParser):
    def _sub_names(self, msg: str, user_name: str, char_name: str) -> str:
        return msg.replace("{{char}}", char_name).replace("{{user}}", user_name)

    def _parse_defs(self, definitions: str) -> List[ChatPrompt]:
        if not definitions.strip():
            return []

        msg_start_re =  re.compile(r"({{random_user_[0-9]+}}|{{user}}|{{char}}):")
        raw_examples = definitions.split("END_OF_DIALOG")
        examples = []
        for raw_ex in raw_examples:
            chunks = msg_start_re.split(raw_ex)[1:]
            messages = [
                ChatMessage(
                    MessageSender.model if c == "{{char}}" else MessageSender.human,
                    text=text.strip(),
                )
                for (c, text) in zip(chunks[::1], chunks[1::2])
            ]
            examples.append(ChatPrompt(messages))
        return examples

    def parse(self, prompt: Dict[str, Any]) -> Prompt:
        bot_char = RoleplayCharacter(
            prompt["bot_name"],
            prompt["bot_description"],
            example_chats=self._parse_defs(prompt["bot_definitions"]),
        )
        messages = []
        for msg in prompt["conversation"]:
            sender = MessageSender.human if msg["is_human"] else MessageSender.model
            text = msg["message"]
            messages.append(ChatMessage(sender, text))
        return RoleplayPrompt(messages, bot_char)


def get_parser(type_: str) -> PromptParser:
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
    elif type_ == "wikitext_document":
        return CompletionParser(key="page")
    elif type_ == "dolphin":
        return OrcaStyleParser.dolphin()
    elif type_ == "openorca":
        return OrcaStyleParser.open_orca()
    elif type_ == "orca_mini":
        return OrcaStyleParser.orca_mini()
    elif type_ == "dolly":
        return GenericInstructParser.dolly()
    elif type_ == "rp_forum":
        return RoleplayForumParser()
    elif type_ == "pippa":
        return PippaParser()
    else:
        raise RuntimeError(f"Unknown parser type: {type_}")
