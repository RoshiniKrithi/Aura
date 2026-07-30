"""PyTest suite for CharacterTokenizer and Vocabulary components."""

import pytest

from src.tokenizer.char_tokenizer import CharacterTokenizer
from src.tokenizer.exceptions import TokenizerError, VocabularyError
from src.tokenizer.vocab import SpecialTokens, Vocabulary


def test_vocabulary_build_and_stats():
    """Verify vocabulary building, character frequency counting, and statistics."""
    corpus = "def add(a, b):\n    return a + b\n"
    vocab = Vocabulary.build_from_corpus(corpus)

    assert len(vocab) > 4  # Special tokens + unique characters
    assert vocab.special_tokens.pad_token in vocab
    assert vocab.special_tokens.unk_token in vocab
    assert vocab.special_tokens.bos_token in vocab
    assert vocab.special_tokens.eos_token in vocab

    stats = vocab.get_stats()
    assert stats["vocab_size"] == len(vocab)
    assert stats["total_character_count"] == len(corpus)
    assert len(stats["top_10_characters"]) <= 10


def test_vocabulary_json_serialization(temp_dir):
    """Verify vocabulary saving to JSON and loading back."""
    corpus = "hello world python"
    vocab1 = Vocabulary.build_from_corpus(corpus)

    save_path = temp_dir / "vocab.json"
    vocab1.save(save_path)
    assert save_path.exists()

    vocab2 = Vocabulary.load(save_path)
    assert len(vocab1) == len(vocab2)
    assert vocab1.char2id == vocab2.char2id
    assert vocab1.special_tokens == vocab2.special_tokens


def test_character_tokenizer_encode_decode_roundtrip():
    """Verify single string encoding and decoding round-trip consistency."""
    corpus = "def fibonacci(n):\n    if n <= 1: return n\n"
    tokenizer = CharacterTokenizer.from_corpus(corpus)

    original_text = "def fibonacci(n):"
    encoded = tokenizer.encode(original_text)
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, list)
    assert all(isinstance(t, int) for t in encoded)
    assert decoded == original_text


def test_character_tokenizer_special_tokens():
    """Verify BOS, EOS, UNK, and PAD token wrapping."""
    corpus = "abc"
    tokenizer = CharacterTokenizer.from_corpus(corpus)

    encoded = tokenizer.encode("ab", add_special_tokens=True)
    assert encoded[0] == tokenizer.vocab.bos_id
    assert encoded[-1] == tokenizer.vocab.eos_id

    decoded_with_special = tokenizer.decode(encoded, skip_special_tokens=False)
    assert "<|startoftext|>" in decoded_with_special
    assert "<|endoftext|>" in decoded_with_special

    decoded_clean = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded_clean == "ab"


def test_unknown_character_handling():
    """Verify unseen characters are substituted with UNK token ID."""
    corpus = "abc"
    tokenizer = CharacterTokenizer.from_corpus(corpus)

    encoded = tokenizer.encode("a z b")  # 'z' is unknown
    assert tokenizer.vocab.unk_id in encoded

    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    assert "<|unk|>" in decoded


def test_batch_encoding_padding_and_truncation():
    """Verify batch encoding with dynamic padding and max_length truncation."""
    corpus = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharacterTokenizer.from_corpus(corpus)

    texts = ["abc", "abcdef", "a"]
    batch_padded = tokenizer.encode_batch(texts, padding=True)

    assert len(batch_padded) == 3
    # All sequences in batch should be padded to longest sequence ("abcdef" -> length 6)
    assert all(len(seq) == 6 for seq in batch_padded)
    assert batch_padded[0][3:] == [tokenizer.vocab.pad_id] * 3

    # Batch with max_length and truncation
    batch_truncated = tokenizer.encode_batch(texts, max_length=4, padding=True, truncation=True)
    assert all(len(seq) == 4 for seq in batch_truncated)


def test_batch_decoding():
    """Verify batch decoding of 2D token lists."""
    corpus = "hello world"
    tokenizer = CharacterTokenizer.from_corpus(corpus)

    texts = ["hello", "world"]
    encoded_batch = tokenizer.encode_batch(texts)
    decoded_batch = tokenizer.decode_batch(encoded_batch)

    assert decoded_batch == texts


def test_edge_case_empty_string():
    """Verify behavior on empty strings."""
    tokenizer = CharacterTokenizer.from_corpus("abc")
    encoded = tokenizer.encode("")
    assert encoded == []

    decoded = tokenizer.decode([])
    assert decoded == ""


def test_edge_case_large_input():
    """Verify performance and correctness on large input text (100,000 characters)."""
    large_text = "def solve():\n    return 42\n" * 4000
    tokenizer = CharacterTokenizer.from_corpus("def solve():\n return 42")

    encoded = tokenizer.encode(large_text)
    assert len(encoded) == len(large_text)

    decoded = tokenizer.decode(encoded)
    assert decoded == large_text


def test_edge_case_repeated_characters():
    """Verify repeated single character sequence handling."""
    tokenizer = CharacterTokenizer.from_corpus("a")
    text = "a" * 100
    encoded = tokenizer.encode(text)

    assert len(encoded) == 100
    assert len(set(encoded)) == 1
    assert tokenizer.decode(encoded) == text


def test_edge_case_unicode_characters():
    """Verify handling of complex Unicode characters and emojis."""
    corpus = "def search(): return '🚀 Python 3.12 🐍'"
    tokenizer = CharacterTokenizer.from_corpus(corpus)

    text = "🚀 Python 3.12 🐍"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_tokenizer_error_exceed_max_length():
    """Verify TokenizerError raised when max_length is exceeded without truncation."""
    tokenizer = CharacterTokenizer.from_corpus("abc")
    with pytest.raises(TokenizerError):
        tokenizer.encode("abcde", max_length=3, truncation=False)


def test_vocabulary_error_invalid_load_path():
    """Verify VocabularyError raised for invalid json load path."""
    with pytest.raises(VocabularyError):
        Vocabulary.load("non_existent_vocab.json")
