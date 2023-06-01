from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TypeAlias, Union


class MessageSender(Enum):
    """Enumerated type representing different types of message senders.

    :cvar human: A string representing a human sender (user).
    :cvar model: A string representing an AI model sender (assistant).
    :cvar system: A string representing a system sender.
    """

    human = "user"
    model = "assistant"
    system = "system"


@dataclass
class ChatMessage:
    """Represents a single message in a conversation.

    If text is None, then this indicates a message is to be generated here.
    """

    sender: MessageSender
    text: Optional[str]


@dataclass
class ChatPrompt:
    """Represents a chat prompt as a list of messages."""

    messages: List[ChatMessage]

    def as_chat(self) -> "ChatPrompt":
        return self

    def as_instruct(self) -> "InstructPrompt":
        """Converts this chat prompt to an equivalent instruct prompt.

        This can only be done successfully for a chat with exactly two messages, an
        initial message from the user and a response from the model.

        :raises RuntimeError: If the conditions for conversion are not satisfied.
        """
        if (
            len(self.messages) != 2
            or self.messages[0].sender != MessageSender.human
            or self.messages[1].sender != MessageSender.model
        ):
            raise RuntimeError("Can't convert this chat prompt to an instruct prompt")
        return InstructPrompt(self.messages[0].text, self.messages[1].text)


@dataclass
class InstructPrompt:
    """A prompt consisting of a single instruction, an optional input, and an output."""

    instruction: str
    output: Optional[str] = None
    input: Optional[str] = None

    def as_instruct(self) -> "InstructPrompt":
        return self

    def as_chat(self) -> ChatPrompt:
        """Convert this instruction to an equivalent chat prompt."""
        message = self.instruction
        if self.input:
            message = f"{message}\n{self.input}"
        return ChatPrompt(
            messages=[
                ChatMessage(MessageSender.human, message),
                ChatMessage(MessageSender.model, self.output),
            ]
        )


CompletionPrompt: TypeAlias = str
"""A section of raw text to be predicted."""

Prompt: TypeAlias = Union[InstructPrompt, ChatPrompt, CompletionPrompt]
"""Represents a value that could be any of the prompt types supported.

For details, see :class: `.InstructPrompt`, :class: `.ChatPrompt`, and
:class: `.CompletionPrompt`.
"""
