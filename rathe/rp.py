from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from rathe.conversion import ConversionContext, PromptConverter
from rathe.formatting import FormatResult, PromptFormatter
from rathe.prompt import ChatMessage, ChatPrompt, MessageSender, Prompt


@dataclass
class RoleplayCharacter:
    name: str
    description: Optional[str] = None
    example_chats: List[ChatPrompt] = field(default_factory=list)


@dataclass
class RoleplayPrompt:
    """A chat prompt augmented with a scenario and character descriptions."""

    messages: List[ChatMessage]
    model_char: RoleplayCharacter
    user_char: Optional[RoleplayCharacter] = None
    context: Optional[str] = None


@dataclass
class RoleplayToChat(PromptConverter[RoleplayPrompt, ChatPrompt]):
    system_format: str = (
        "You are roleplaying as {bot.name}.\n{bot.name}'s persona:\n{bot.description}"
    )

    def convert(self, prompt: RoleplayPrompt) -> ChatPrompt:
        system_message = self.system_format.format(
            bot=prompt.model_char, human=prompt.user_char, context=prompt.context
        )
        return ChatPrompt(
            messages=[ChatMessage(MessageSender.system, system_message)]
            + prompt.messages
        )


@dataclass
class GuiseFormatter(PromptFormatter):
    def format(
        self,
        prompt: Prompt,
        special_tokens: Dict[str, str],
        conversion_context: ConversionContext | None = None,
    ) -> FormatResult:
        prompt: RoleplayPrompt = conversion_context.convert(prompt, RoleplayPrompt)
        res = FormatResult()

        res.add(f"<|character|>{prompt.model_char.name}", is_input=True)
        if prompt.model_char.description:
            res.add(f"<|bio|>{prompt.model_char.description}", is_input=True)

        if prompt.model_char.example_chats:
            res.add("<|examples|>", is_input=True)
            for chat in prompt.model_char.example_chats:
                for msg in chat.messages:
                    tag = "<|user|>" if msg.sender == MessageSender.human else "<|bot|>"
                    res.add(f"{tag}{msg.text}", is_input=True)
                res.add("<|eoc|>", is_input=True)

        res.add("<|history|>")
        for msg in prompt.messages[:-1]:
            tag = "<|user|>" if msg.sender == MessageSender.human else "<|bot|>"
            res.add(f"{tag}{msg.text}", is_input=True)
        res.add("<|response|>", is_input=True)

        res.add(prompt.messages[-1].text, is_input=False)
        return res
