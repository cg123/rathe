import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rathe.conversion import ConversionContext, PromptConverter
from rathe.formatting import ChatPromptFormatter, FormatResult, PromptFormatter
from rathe.prompt import ChatMessage, ChatPrompt, MessageSender, Prompt


@dataclass
class RoleplayCharacter:
    name: str
    description: Optional[str] = None
    example_chats: List[ChatPrompt] = field(default_factory=list)


@dataclass
class RoleplayMessage:
    sender: str
    text: str


@dataclass
class RoleplayPrompt:
    """A chat prompt augmented with a scenario and character descriptions."""

    messages: List[RoleplayMessage]
    model_char: RoleplayCharacter
    user_char: Optional[RoleplayCharacter] = None
    context: Optional[str] = None


@dataclass
class RoleplayToChat(PromptConverter[RoleplayPrompt, ChatPrompt]):
    system_format: str = (
        "You are roleplaying as {bot.name}.\n{bot.name}'s persona:\n{bot.description}"
    )

    def _convert_message(
        self, msg: RoleplayMessage, prompt: RoleplayPrompt
    ) -> ChatMessage:
        if msg.sender == prompt.model_char.name:
            return ChatMessage(MessageSender.model, msg.text)
        sender_prefix = ""
        if msg.sender in ["", "user", "human"]:
            if prompt.user_char:
                sender_prefix = prompt.user_char.name + ": "
            else:
                sender_prefix = ""
        else:
            sender_prefix = msg.sender + ": "

        return ChatMessage(MessageSender.human, sender_prefix + msg.text)

    def convert(self, prompt: RoleplayPrompt) -> ChatPrompt:
        system_message = self.system_format.format(
            bot=prompt.model_char, human=prompt.user_char, context=prompt.context
        )
        return ChatPrompt(
            messages=[ChatMessage(MessageSender.system, system_message)]
            + [
                self._convert_message(msg, prompt.model_char.name)
                for msg in prompt.messages
            ]
        )


@dataclass
class LengthRangeDescriptor:
    min_words: int
    max_words: int
    text: str


DEFAULT_LENGTH_DESCRIPTORS = [
    LengthRangeDescriptor(*tup)
    for tup in [
        (0, 30, "microscopic"),  # 3 tokens
        (0, 85, "tiny"),
        (40, 80, "quite short"),  # 2 tokens
        (50, 110, "concise"),  # 2 tokens
        (85, 130, "short"),
        (100, 160, "brief"),
        (130, 180, "medium"),
        (160, 220, "moderate"),  # 2 tokens
        (150, 200, "somewhat long"),  # 2 tokens
        (180, 240, "long"),
        (210, 280, "lengthy"),  # 2 tokens
        (220, 290, "extended"),
        (240, 305, "very long"),  # 2 tokens
        (305, 390, "humongous"),  # 3 tokens
        (350, 410, "hundreds of words"),  # 3 tokens
        (360, 450, "comprehensive"),  # 2 tokens
        (390, 500, "extremely long"),  # 2 tokens
    ]
]


def describe_length(text: str, probability: float = 1.0) -> Optional[str]:
    word_count = len(text.split())

    if probability < 1 and random.random() > probability:
        return None

    applicable = [
        d
        for d in DEFAULT_LENGTH_DESCRIPTORS
        if word_count >= d.min_words and word_count < d.max_words
    ]
    if not applicable:
        return None

    return random.choice(applicable).text


@dataclass
class ChatMlRpFormatter(PromptFormatter):
    inner: ChatPromptFormatter
    system_format: str = (
        "Enter roleplay mode. You are {model_char.name}.\n{model_char.description}"
    )
    length_annotate_prob: float = 0.5

    def _message_wrap(self, sender: str, text: str) -> str:
        return f"<|im_start|>{sender}\n{text}<|im_end|>\n"

    def _ex_sender_name(self, msg: ChatMessage, bot_name: str) -> str:
        if msg.sender == MessageSender.human:
            return "User"
        return bot_name

    def format(
        self,
        prompt: Prompt,
        special_tokens: Dict[str, str],
        conversion_context: Optional[ConversionContext] = None,
    ) -> FormatResult:
        if conversion_context is None:
            conversion_context = ConversionContext.default()

        if not isinstance(prompt, RoleplayPrompt):
            return self.inner.format(prompt, special_tokens, conversion_context)

        res = FormatResult()
        res.add(
            self._message_wrap(
                sender="system",
                text=self.system_format.format(model_char=prompt.model_char),
            ),
            is_input=True,
        )
        if prompt.model_char.example_chats:
            for idx, chat in enumerate(prompt.model_char.example_chats):
                log = "\n".join(
                    f"{self._ex_sender_name(msg, prompt.model_char.name)}: {msg.text}"
                    for msg in chat.messages
                )
                res.add(
                    self._message_wrap(
                        sender="system", text=f"Example session #{idx + 1}:\n{log}"
                    ),
                    is_input=True,
                )

        for msg in prompt.messages:
            sender = msg.sender
            if self.length_annotate_prob > 0 and sender == prompt.model_char.name:
                length_text = describe_length(msg.text, self.length_annotate_prob)
                if length_text:
                    sender = f"{sender} (Length: {length_text})"

            res.add(
                self._message_wrap(sender, msg.text),
                is_input=msg.sender != prompt.model_char.name,
            )

        return res


@dataclass
class InstructRpFormatter(PromptFormatter):
    inner: PromptFormatter
    system_format: str = (
        "Enter roleplay mode. You are {model_char.name}.\n{model_char.description}\n"
    )
    length_annotate_prob: float = 0.5

    def _ex_sender_name(self, msg: ChatMessage, bot_name: str) -> str:
        if msg.sender == MessageSender.human:
            return "User"
        return bot_name

    def format(
        self,
        prompt: Prompt,
        special_tokens: Dict[str, str],
        conversion_context: Optional[ConversionContext] = None,
    ) -> FormatResult:
        if conversion_context is None:
            conversion_context = ConversionContext.default()

        if not isinstance(prompt, RoleplayPrompt):
            return self.inner.format(prompt, special_tokens, conversion_context)

        user_name = prompt.user_char.name if prompt.user_char else "User"
        while prompt.messages and prompt.messages[-1].sender in (
            MessageSender.human,
            user_name,
        ):
            prompt.messages.pop(-1)

        if not prompt.messages:
            return FormatResult([])

        instruction = self.system_format.format(model_char=prompt.model_char)

        if prompt.model_char.example_chats:
            chunks = []
            for idx, chat in enumerate(prompt.model_char.example_chats):
                log = "\n".join(
                    f"{self._ex_sender_name(msg, prompt.model_char.name)}: {msg.text}"
                    for msg in chat.messages
                )
                if not log.strip():
                    continue
                chunks.append(f"Example session #{idx + 1}:\n```\n{log}```\n")
            instruction += "\n".join(chunks)

        res = FormatResult()
        res.add(
            f"### Instruction:{chr(10) * random.choice([1, 1, 2])}{instruction}{chr(10) * random.choice([1, 2, 2])}",
            is_input=True,
        )

        if prompt.messages[:-1]:
            res.add(f"### Input:{chr(10) * random.choice([1, 1, 2])}", is_input=True)
            for msg in prompt.messages[:-1]:
                sender_name = self._ex_sender_name(msg, prompt.model_char.name)
                res.add(f"{sender_name}:", is_input=True)
                res.add(
                    f"{msg.text}\n",
                    is_input=msg.sender != prompt.model_char.name,
                )

        last_msg = prompt.messages[-1]
        length_suffix = ""
        if self.length_annotate_prob > 0:
            length_text = describe_length(last_msg.text, self.length_annotate_prob)
            if length_text:
                length_suffix = f" (Length: {length_text})"

        res.add(
            f"{chr(10) * random.choice([0, 1, 2])}### Response{length_suffix}:{chr(10) * random.choice([1, 1, 2])}",
            is_input=True,
        )
        sender_name = self._ex_sender_name(last_msg, prompt.model_char.name)
        res.add(f"{sender_name}:", is_input=True)
        res.add(last_msg.text, is_input=False)
        return res


class ChaiTruncatingFormatter(PromptFormatter):
    do_truncate: bool = True

    def _ex_sender_name(self, msg: ChatMessage, bot_name: str) -> str:
        if msg.sender == MessageSender.human:
            return "User"
        return bot_name

    def format(
        self,
        prompt: Prompt,
        special_tokens: Dict[str, str],
        conversion_context: ConversionContext | None = None,
    ) -> FormatResult:
        if conversion_context is None:
            conversion_context = ConversionContext.default()

        if not isinstance(prompt, RoleplayPrompt):
            raise NotImplementedError()

        if "**<MEMORY>**" in prompt.model_char.description:
            chai_prompt, chai_memory = prompt.model_char.description.split(
                "**<MEMORY>**"
            )
        else:
            chai_prompt = prompt.model_char.description
            if prompt.model_char.example_chats:
                chunks = []
                for idx, chat in enumerate(prompt.model_char.example_chats):
                    log = "\n".join(
                        f"{self._ex_sender_name(msg, prompt.model_char.name)}: {msg.text}"
                        for msg in chat.messages
                    )
                    if not log.strip():
                        continue
                    chunks.append(log)
                chai_memory = "\n\n".join(chunks)
            else:
                chai_memory = ""

        if self.do_truncate:
            # yes, character-wise
            chai_memory = chai_memory[:1024]
            chai_prompt = chai_prompt[:1024]

        if prompt.user_char:
            user_name = prompt.user_char.name
        else:
            user_name = "Anonymous user"

        header = f" ***character:{prompt.model_char.name} ***description:{chai_prompt} ***memory:{chai_memory} ***username:{user_name}"
        res = FormatResult()
        res.add(header, is_input=True)

        for msg in prompt.messages:
            is_user = msg.sender != prompt.model_char.name
            tag = " ***user:" if is_user else f" ***agent:"
            res.add(tag, is_input=True)
            res.add(f"{msg.sender}:{msg.text}", is_input=not is_user)
        return res
