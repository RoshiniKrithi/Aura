"""Conversation Formatter, Prompt Template Engine, and Token Masking for EXP-004 Instruction Tuning.

Provides ChatML template formatting, role delimitation, conversation tokenization,
and completion-only target loss masking (-100 PyTorch ignore_index strategy).
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from src.tokenizer.code_bpe_tokenizer import CodeBPETokenizer

logger = logging.getLogger(__name__)

# Special ChatML Control Tokens
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|endoftext|>"

IM_START_ID = 50257
IM_END_ID = 50258
PAD_ID = 50259
EOS_ID = 3


@dataclass
class Message:
    """Represents a single message turn in a conversation.

    Attributes:
        role: Message sender role ("system", "user", "assistant").
        content: Text content of the message turn.
    """

    role: str
    content: str


@dataclass
class Conversation:
    """Represents a full multi-turn instruction conversation sequence.

    Attributes:
        messages: List of Message turn objects.
        system_prompt: Optional custom system prompt override.
        metadata: Optional metadata dictionary (domain, dataset_source, complexity).
    """

    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptTemplateEngine:
    """Formats conversation objects into ChatML prompt text representations."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are Aura, an expert AI assistant specializing in Programming, Data Structures, "
        "Algorithms, and Software Engineering."
    )

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        """Initializes PromptTemplateEngine.

        Args:
            system_prompt: Custom default system prompt.
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def format(self, conversation: Conversation) -> str:
        """Renders conversation into canonical ChatML format string.

        Args:
            conversation: Conversation container.

        Returns:
            Formatted ChatML text string.
        """
        formatted_parts = []
        sys_prompt = conversation.system_prompt or self.system_prompt

        if sys_prompt:
            formatted_parts.append(f"{IM_START_TOKEN}system\n{sys_prompt}{IM_END_TOKEN}\n")

        for msg in conversation.messages:
            formatted_parts.append(f"{IM_START_TOKEN}{msg.role}\n{msg.content}{IM_END_TOKEN}\n")

        return "".join(formatted_parts)


class ConversationTokenizer:
    """Encodes conversation text streams and handles special control token IDs."""

    def __init__(self, tokenizer: Optional[CodeBPETokenizer] = None) -> None:
        """Initializes ConversationTokenizer.

        Args:
            tokenizer: Optional pre-configured CodeBPETokenizer instance.
        """
        self.tokenizer = tokenizer or CodeBPETokenizer.create_default()

    def encode_text(self, text: str) -> List[int]:
        """Encodes raw text into token ID list.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_ids(self, ids: List[int]) -> str:
        """Decodes token ID list back into text string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        return self.tokenizer.decode(ids)


class ConversationFormatter:
    """Converts Conversation objects into PyTorch token tensors with completion loss masking."""

    def __init__(
        self,
        tokenizer: Optional[CodeBPETokenizer] = None,
        template_engine: Optional[PromptTemplateEngine] = None,
        ignore_index: int = -100,
    ) -> None:
        """Initializes ConversationFormatter.

        Args:
            tokenizer: Optional CodeBPETokenizer instance.
            template_engine: Optional PromptTemplateEngine instance.
            ignore_index: PyTorch loss ignore index (default -100).
        """
        self.tokenizer = tokenizer or CodeBPETokenizer.create_default()
        self.conv_tokenizer = ConversationTokenizer(self.tokenizer)
        self.template_engine = template_engine or PromptTemplateEngine()
        self.ignore_index = ignore_index

    def tokenize_and_mask(
        self,
        conversation: Conversation,
        max_sequence_length: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes conversation and generates input_ids (X) and label targets (Y).

        Loss is calculated ONLY on Assistant response tokens. System prompts, user queries,
        and ChatML control headers are assigned label -100 (ignored during loss reduction).

        Args:
            conversation: Conversation object.
            max_sequence_length: Maximum sequence context length L.

        Returns:
            Tuple of (input_ids, labels) LongTensors of shape (seq_len,).
        """
        sys_prompt = conversation.system_prompt or self.template_engine.system_prompt

        full_token_ids: List[int] = []
        label_ids: List[int] = []

        # 1. System Prompt
        if sys_prompt:
            sys_str = f"{IM_START_TOKEN}system\n{sys_prompt}{IM_END_TOKEN}\n"
            sys_tokens = [IM_START_ID] + self.conv_tokenizer.encode_text(f"system\n{sys_prompt}") + [IM_END_ID]
            full_token_ids.extend(sys_tokens)
            label_ids.extend([self.ignore_index] * len(sys_tokens))

        # 2. Conversation Turns
        for msg in conversation.messages:
            role = msg.role.strip().lower()
            content = msg.content.strip()

            header_str = f"{IM_START_TOKEN}{role}\n"
            header_tokens = [IM_START_ID] + self.conv_tokenizer.encode_text(f"{role}\n")

            content_tokens = self.conv_tokenizer.encode_text(content)
            end_tokens = [IM_END_ID]

            turn_tokens = header_tokens + content_tokens + end_tokens
            full_token_ids.extend(turn_tokens)

            if role == "assistant":
                # Header tokens are ignored, content + END tokens are trained
                turn_labels = ([self.ignore_index] * len(header_tokens)) + content_tokens + end_tokens
            else:
                # System or User tokens are completely ignored
                turn_labels = [self.ignore_index] * len(turn_tokens)

            label_ids.extend(turn_labels)

        # Truncate if exceeding max_sequence_length
        if len(full_token_ids) > max_sequence_length:
            full_token_ids = full_token_ids[:max_sequence_length]
            label_ids = label_ids[:max_sequence_length]

        x_ids = full_token_ids
        y_ids = label_ids

        # Pad to max_sequence_length
        pad_len = max_sequence_length - len(x_ids)
        if pad_len > 0:
            x_ids = x_ids + [PAD_ID] * pad_len
            y_ids = y_ids + [self.ignore_index] * pad_len

        x_tensor = torch.tensor(x_ids, dtype=torch.long)
        y_tensor = torch.tensor(y_ids, dtype=torch.long)

        return x_tensor, y_tensor
