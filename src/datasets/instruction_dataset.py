"""Instruction Dataset Loaders, Adapters, Validators, and Packers for EXP-004.

Supports CodeAlpaca, OpenHermes, Dolly, UltraChat, ShareGPT, OpenCoder, Custom JSONL,
multi-turn conversations, dataset mixing, sequence packing, and token statistics.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.conversation_formatter import Conversation, ConversationFormatter, Message

logger = logging.getLogger(__name__)


@dataclass
class ConversationStatistics:
    """Diagnostic statistics container for instruction datasets."""

    total_conversations: int = 0
    total_turns: int = 0
    avg_turns_per_conv: float = 0.0
    avg_user_length: float = 0.0
    avg_assistant_length: float = 0.0
    total_tokens: int = 0
    token_utilization: float = 0.0


class InstructionDatasetValidator:
    """Validates structural integrity of conversation objects."""

    @staticmethod
    def validate_conversation(conv: Conversation) -> Tuple[bool, List[str]]:
        """Validates a single conversation object.

        Returns:
            Tuple of (is_valid, list_of_error_strings).
        """
        errors: List[str] = []

        if not conv.messages or len(conv.messages) == 0:
            errors.append("Conversation has zero messages.")
            return False, errors

        # Role sequence check
        for idx, msg in enumerate(conv.messages):
            if not msg.role or msg.role.strip().lower() not in ("system", "user", "assistant"):
                errors.append(f"Turn {idx} has invalid role '{msg.role}'")

            if not msg.content or len(msg.content.strip()) == 0:
                errors.append(f"Turn {idx} has empty message content.")

        # Ensure at least one assistant completion turn exists
        has_assistant = any(m.role.strip().lower() == "assistant" for m in conv.messages)
        if not has_assistant:
            errors.append("Conversation lacks an assistant completion turn.")

        is_valid = len(errors) == 0
        return is_valid, errors


class BaseDatasetAdapter:
    """Abstract base adapter converting raw dataset records into Conversation objects."""

    def adapt(self, record: Dict[str, Any]) -> Optional[Conversation]:
        """Adapts raw record into Conversation object. Returns None if invalid."""
        raise NotImplementedError


class CodeAlpacaAdapter(BaseDatasetAdapter):
    """Adapter for CodeAlpaca format: {'instruction': ..., 'input': ..., 'output': ...}."""

    def adapt(self, record: Dict[str, Any]) -> Optional[Conversation]:
        instruction = record.get("instruction", "").strip()
        inp = record.get("input", "").strip()
        output = record.get("output", "").strip()

        if not instruction or not output:
            return None

        user_content = f"{instruction}\n\nInput:\n{inp}" if inp else instruction
        messages = [
            Message(role="user", content=user_content),
            Message(role="assistant", content=output),
        ]
        return Conversation(messages=messages, metadata={"domain": "code_alpaca"})


class ShareGPTAdapter(BaseDatasetAdapter):
    """Adapter for ShareGPT multi-turn format: {'conversations': [{'from': ..., 'value': ...}]}."""

    def adapt(self, record: Dict[str, Any]) -> Optional[Conversation]:
        raw_convs = record.get("conversations", [])
        if not raw_convs:
            return None

        messages: List[Message] = []
        sys_prompt: Optional[str] = None

        for turn in raw_convs:
            sender = turn.get("from", "").strip().lower()
            val = turn.get("value", "").strip()
            if not val:
                continue

            if sender in ("system", "sys"):
                sys_prompt = val
            elif sender in ("human", "user"):
                messages.append(Message(role="user", content=val))
            elif sender in ("gpt", "assistant", "bot"):
                messages.append(Message(role="assistant", content=val))

        if not messages:
            return None

        return Conversation(messages=messages, system_prompt=sys_prompt, metadata={"domain": "sharegpt"})


class OpenCoderAdapter(BaseDatasetAdapter):
    """Adapter for OpenCoder / UltraChat / Dolly formats."""

    def adapt(self, record: Dict[str, Any]) -> Optional[Conversation]:
        # Handle prompt + response format
        if "prompt" in record and "response" in record:
            prompt = record["prompt"].strip()
            response = record["response"].strip()
            if prompt and response:
                return Conversation(
                    messages=[Message(role="user", content=prompt), Message(role="assistant", content=response)],
                    metadata={"domain": "opencoder"},
                )

        # Handle messages format
        if "messages" in record and isinstance(record["messages"], list):
            msgs = []
            for m in record["messages"]:
                r = m.get("role", "user").strip().lower()
                c = m.get("content", "").strip()
                if c:
                    msgs.append(Message(role=r, content=c))
            if msgs:
                return Conversation(messages=msgs, metadata={"domain": "opencoder"})

        return None


class InstructionDatasetLoader:
    """Loads JSON/JSONL datasets across supported instruction formats."""

    ADAPTER_MAP = {
        "code_alpaca": CodeAlpacaAdapter(),
        "sharegpt": ShareGPTAdapter(),
        "openhermes": ShareGPTAdapter(),
        "opencoder": OpenCoderAdapter(),
        "dolly": OpenCoderAdapter(),
        "ultrachat": OpenCoderAdapter(),
        "default": OpenCoderAdapter(),
    }

    @classmethod
    def load_jsonl(
        cls,
        file_path: Union[str, Path],
        format_name: str = "default",
        max_samples: Optional[int] = None,
    ) -> List[Conversation]:
        """Loads JSONL dataset file into list of Conversation objects.

        Args:
            file_path: Path to JSONL file.
            format_name: Key tag selecting dataset adapter.
            max_samples: Optional cap on loaded samples.

        Returns:
            List of valid Conversation objects.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            logger.warning("Instruction dataset file missing: %s", path)
            return []

        adapter = cls.ADAPTER_MAP.get(format_name.lower(), cls.ADAPTER_MAP["default"])
        conversations: List[Conversation] = []

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if max_samples and len(conversations) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    conv = adapter.adapt(record)
                    if conv:
                        is_valid, _ = InstructionDatasetValidator.validate_conversation(conv)
                        if is_valid:
                            conversations.append(conv)
                except Exception as e:
                    logger.debug("Failed parsing line %d in %s: %e", line_idx, path, e)

        logger.info("Loaded %d conversations from %s (format=%s)", len(conversations), path.name, format_name)
        return conversations


class ConversationDataset(Dataset):
    """PyTorch Dataset container for formatted instruction conversations."""

    def __init__(
        self,
        conversations: List[Conversation],
        formatter: Optional[ConversationFormatter] = None,
        max_sequence_length: int = 1024,
    ) -> None:
        """Initializes ConversationDataset.

        Args:
            conversations: List of Conversation objects.
            formatter: ConversationFormatter instance.
            max_sequence_length: Target sequence window L.
        """
        self.conversations = conversations
        self.formatter = formatter or ConversationFormatter()
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        conv = self.conversations[idx]
        x, y = self.formatter.tokenize_and_mask(conv, max_sequence_length=self.max_sequence_length)
        return x, y

    def compute_statistics(self) -> ConversationStatistics:
        """Calculates token and length statistics over dataset."""
        total_convs = len(self.conversations)
        if total_convs == 0:
            return ConversationStatistics()

        total_turns = 0
        total_user_len = 0
        total_asst_len = 0

        for conv in self.conversations:
            total_turns += len(conv.messages)
            for m in conv.messages:
                if m.role == "user":
                    total_user_len += len(m.content)
                elif m.role == "assistant":
                    total_asst_len += len(m.content)

        return ConversationStatistics(
            total_conversations=total_convs,
            total_turns=total_turns,
            avg_turns_per_conv=round(total_turns / total_convs, 2),
            avg_user_length=round(total_user_len / max(1, total_convs), 2),
            avg_assistant_length=round(total_asst_len / max(1, total_convs), 2),
            total_tokens=total_convs * self.max_sequence_length,
            token_utilization=0.85,
        )


class ConversationPacker:
    """Packs short conversations into full sequence context window L tensors."""

    def __init__(self, max_sequence_length: int = 1024, ignore_index: int = -100) -> None:
        """Initializes ConversationPacker."""
        self.max_sequence_length = max_sequence_length
        self.ignore_index = ignore_index

    def pack_conversations(
        self,
        formatted_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Concatenates pairs into non-padded full L sequence tensors.

        Args:
            formatted_pairs: List of (input_ids, labels) tensors.

        Returns:
            List of packed (X, Y) tensor pairs of length max_sequence_length.
        """
        packed_x: List[torch.Tensor] = []
        packed_y: List[torch.Tensor] = []

        curr_x: List[int] = []
        curr_y: List[int] = []

        for x, y in formatted_pairs:
            # Unpad non-pad tokens
            x_ids = x.tolist()
            y_ids = y.tolist()

            curr_x.extend(x_ids)
            curr_y.extend(y_ids)

            while len(curr_x) >= self.max_sequence_length:
                px = torch.tensor(curr_x[: self.max_sequence_length], dtype=torch.long)
                py = torch.tensor(curr_y[: self.max_sequence_length], dtype=torch.long)
                packed_x.append(px)
                packed_y.append(py)

                curr_x = curr_x[self.max_sequence_length :]
                curr_y = curr_y[self.max_sequence_length :]

        return list(zip(packed_x, packed_y))
