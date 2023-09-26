from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from rathe.conversion import PromptConverter
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
