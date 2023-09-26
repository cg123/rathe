from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Tuple, Type

from typing_extensions import TypeVar

from rathe.prompt import ChatMessage, ChatPrompt, InstructPrompt, MessageSender

SourceT = TypeVar("SourceT")
TargetT = TypeVar("TargetT")


class PromptConverter(ABC, Generic[SourceT, TargetT]):
    @abstractmethod
    def convert(self, prompt: SourceT) -> TargetT:
        ...

    def can_convert(self, prompt: SourceT) -> bool:
        return True


class InstructToChat(PromptConverter[InstructPrompt, ChatPrompt]):
    def convert(self, prompt: InstructPrompt) -> ChatPrompt:
        message = prompt.instruction
        if prompt.input:
            message = f"{message}\n{self.input}"
        return ChatPrompt(
            messages=[
                ChatMessage(MessageSender.human, message),
                ChatMessage(MessageSender.model, self.output),
            ]
        )


class SingleTurnChatToInstruct(PromptConverter[ChatPrompt, InstructPrompt]):
    def convert(self, prompt: ChatPrompt) -> InstructPrompt:
        if not self.can_convert(prompt):
            raise RuntimeError("Can't convert this chat prompt to an instruct prompt")
        return InstructPrompt(prompt.messages[0].text, self.messages[1].text)

    def can_convert(self, prompt: ChatPrompt) -> bool:
        return (
            len(prompt.messages) == 2
            and prompt.messages[0].sender == MessageSender.human
            and prompt.messages[1].sender == MessageSender.model
        )


T = TypeVar("T")


class ConversionContext:
    converters: Dict[Tuple[type, type], PromptConverter]

    def __init__(
        self, converters: Optional[Dict[Tuple[type, type], PromptConverter]] = None
    ) -> None:
        if converters is None:
            converters = {}
        self.converters = converters

    @classmethod
    def default(cls) -> "ConversionContext":
        return ConversionContext(
            {
                (InstructPrompt, ChatPrompt): InstructToChat(),
                (ChatPrompt, InstructPrompt): SingleTurnChatToInstruct(),
            }
        )

    def extend(
        self, new_converters: Dict[Tuple[type, type], PromptConverter]
    ) -> "ConversionContext":
        return ConversionContext(converters={**self.converters, **new_converters})

    def convert(self, prompt: Any, target_type: Type[T]) -> T:
        source_type = type(prompt)
        if (source_type, target_type) in self.converters:
            return self.converters[(source_type, target_type)].convert(prompt)
        elif issubclass(source_type, target_type):
            return prompt
        else:
            raise RuntimeError("Unsupported conversion")

    def can_convert(self, prompt: Any, target_type: Type[T]) -> bool:
        source_type = type(prompt)
        if (source_type, target_type) in self.converters:
            return self.converters[(source_type, target_type)].can_convert(prompt)
        elif issubclass(source_type, target_type):
            return True
        return False
