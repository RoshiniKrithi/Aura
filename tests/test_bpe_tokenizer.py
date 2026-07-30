"""PyTest suite for BPETokenizer, BPETrainer, and BPEVocab components."""

import pytest

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.bpe_trainer import BPETrainer
from src.tokenizer.bpe_vocab import BPESpecialTokens, BPEVocab
from src.tokenizer.char_tokenizer import CharacterTokenizer


@pytest.fixture
def python_code_corpus():
    """Sample Python programming & DSA corpus snippet."""
    return """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""


def test_bpe_trainer_and_vocab_creation(python_code_corpus):
    """Verify BPE trainer learns merges up to target vocabulary size."""
    trainer = BPETrainer(vocab_size=80)
    vocab = trainer.train(python_code_corpus)

    assert isinstance(vocab, BPEVocab)
    assert len(vocab) <= 80
    assert len(vocab.merges) > 0
    assert vocab.special_tokens.pad_token in vocab
    assert vocab.special_tokens.unk_token in vocab


def test_bpe_vocab_serialization(python_code_corpus, temp_dir):
    """Verify BPE vocab and merges saving to disk and reloading."""
    trainer = BPETrainer(vocab_size=70)
    vocab1 = trainer.train(python_code_corpus)

    v_file = temp_dir / "vocab.json"
    m_file = temp_dir / "merges.txt"

    vocab1.save(v_file, m_file)
    assert v_file.exists()
    assert m_file.exists()

    vocab2 = BPEVocab.load(v_file, m_file)
    assert len(vocab1) == len(vocab2)
    assert set(vocab1.merges.keys()) == set(vocab2.merges.keys())
    assert vocab1.token2id == vocab2.token2id



def test_bpe_encode_decode_roundtrip(python_code_corpus):
    """Verify BPE encode and decode roundtrip fidelity."""
    trainer = BPETrainer(vocab_size=90)
    vocab = trainer.train(python_code_corpus)
    tokenizer = BPETokenizer(vocab=vocab)

    text = "def binary_search(arr, target):"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, list)
    assert all(isinstance(tid, int) for tid in encoded)
    assert decoded == text


def test_bpe_compression_ratio(python_code_corpus):
    """Verify BPE compresses text sequence length compared to character tokenization."""
    # Build Character Tokenizer
    char_tok = CharacterTokenizer.from_corpus(python_code_corpus)

    # Build BPE Tokenizer
    bpe_trainer = BPETrainer(vocab_size=100)
    bpe_vocab = bpe_trainer.train(python_code_corpus)
    bpe_tok = BPETokenizer(vocab=bpe_vocab)

    sample_text = "def quick_sort(arr): return arr"
    char_encoded = char_tok.encode(sample_text)
    bpe_encoded = bpe_tok.encode(sample_text)

    compression_ratio = bpe_tok.get_compression_ratio(sample_text)

    # BPE token count should be significantly smaller than raw character count
    assert len(bpe_encoded) < len(char_encoded)
    assert compression_ratio > 1.0


def test_bpe_batch_encoding_and_padding(python_code_corpus):
    """Verify BPE batch encoding with dynamic padding."""
    trainer = BPETrainer(vocab_size=80)
    vocab = trainer.train(python_code_corpus)
    tokenizer = BPETokenizer(vocab=vocab)

    texts = ["def binary_search(arr):", "return -1"]
    batch_encoded = tokenizer.encode_batch(texts, padding=True)

    assert len(batch_encoded) == 2
    assert len(batch_encoded[0]) == len(batch_encoded[1])


def test_bpe_edge_cases(python_code_corpus):
    """Verify empty string and special tokens handling in BPE."""
    trainer = BPETrainer(vocab_size=60)
    vocab = trainer.train(python_code_corpus)
    tokenizer = BPETokenizer(vocab=vocab)

    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""

    encoded_sp = tokenizer.encode("def test():", add_special_tokens=True)
    assert encoded_sp[0] == tokenizer.vocab.bos_id
    assert encoded_sp[-1] == tokenizer.vocab.eos_id

    decoded_clean = tokenizer.decode(encoded_sp, skip_special_tokens=True)
    assert decoded_clean == "def test():"
