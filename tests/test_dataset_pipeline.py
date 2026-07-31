"""Comprehensive Unit Tests for Aura Dataset Pipeline.

Validates text cleaning, dataset validation, readers, statistics, vocabulary inspection,
sequence building, caching, PyTorch datasets, splitting, collation, batching, and factory.
"""

from pathlib import Path
import pytest
import torch

from src.datasets import (
    AuraStreamingDataset,
    AuraTextDataset,
    BatchBuilder,
    CollateFunction,
    DatasetCache,
    DatasetFactory,
    DatasetReader,
    DatasetSplitter,
    DatasetStatistics,
    DatasetValidationError,
    DatasetValidator,
    FolderDatasetReader,
    SequenceBuilder,
    StreamingReader,
    TextCleaner,
    TextFileReader,
    VocabularyInspector,
)
from src.tokenizer import CharacterTokenizer
from src.utils.config import (
    BatchConfig,
    CacheConfig,
    DatasetConfig,
    SequenceConfig,
    SplitConfig,
    ValidationConfig,
)


@pytest.fixture
def mock_tokenizer():
    """Returns a CharacterTokenizer instance fitted on standard text."""
    text = "The quick brown fox jumps over the lazy dog 0123456789\n!:;'"
    return CharacterTokenizer.from_corpus(text)


@pytest.fixture
def sample_text_file(tmp_path):
    """Creates a temporary sample text file."""
    p = tmp_path / "sample.txt"
    p.write_text(
        "The quick brown fox jumps over the lazy dog.\nLine 2 of sample text.\nLine 3 of sample text.",
        encoding="utf-8",
    )
    return p


# 1. TextCleaner Tests
def test_text_cleaner():
    cleaner = TextCleaner(
        normalize_unicode="NFC",
        remove_unprintable=True,
        fix_line_endings=True,
        expand_tabs=True,
        tab_size=4,
    )
    raw = "\tHello\r\nWorld\x00null bytes\x07and tabs."
    cleaned = cleaner.clean(raw)
    assert "\r\n" not in cleaned
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "    Hello" in cleaned  # Tab expanded at column 0


# 2. DatasetValidator Tests
def test_dataset_validator_valid_file(sample_text_file):
    validator = DatasetValidator()
    result = validator.validate_file(sample_text_file)
    assert result.is_valid is True
    assert result.total_files == 1
    assert result.valid_files == 1
    assert result.total_bytes > 0


def test_dataset_validator_empty_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    validator = DatasetValidator()
    result = validator.validate_file(empty_file)
    assert result.is_valid is False
    assert any("Empty" in err for err in result.errors)


def test_dataset_validator_invalid_utf8(tmp_path):
    bad_file = tmp_path / "corrupt.bin"
    bad_file.write_bytes(b"\x80\x81\xFF\xFE\xFA")
    validator = DatasetValidator()
    result = validator.validate_file(bad_file)
    assert result.is_valid is False
    assert any("UTF-8" in err for err in result.errors)


def test_dataset_validator_duplicate_detection(tmp_path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    content = "Exact duplicate content across files."
    f1.write_text(content, encoding="utf-8")
    f2.write_text(content, encoding="utf-8")

    validator = DatasetValidator(ValidationConfig(check_duplicates=True))
    result = validator.validate_files([f1, f2])
    assert len(result.duplicate_files) == 1
    assert len(result.warnings) > 0


# 3. Readers Tests
def test_text_file_reader(sample_text_file):
    reader = TextFileReader(sample_text_file)
    all_text = reader.read_all()
    assert "quick brown fox" in all_text

    chunks = list(reader.read_chunks(chunk_size=20))
    assert len(chunks) > 1
    assert "".join(chunks) == all_text


def test_folder_dataset_reader(tmp_path):
    (tmp_path / "a.txt").write_text("File A content.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("File B content.", encoding="utf-8")

    reader = FolderDatasetReader(tmp_path, pattern="*.txt")
    concat = reader.read_all()
    assert "File A content." in concat
    assert "File B content." in concat


def test_streaming_reader(tmp_path):
    f1 = tmp_path / "stream1.txt"
    f2 = tmp_path / "stream2.txt"
    f1.write_text("Line 1\nLine 2\n", encoding="utf-8")
    f2.write_text("Line 3\nLine 4\n", encoding="utf-8")

    s_reader = StreamingReader([f1, f2])
    lines = list(s_reader.read_lines())
    assert len(lines) == 4
    assert lines[0] == "Line 1\n"
    assert lines[2] == "Line 3\n"


# 4. Statistics and Vocab Inspection Tests
def test_dataset_statistics():
    text = "Hello World!\nSecond Line."
    stats = DatasetStatistics.compute_text_stats(text)
    assert stats.total_characters == len(text)
    assert stats.total_lines == 2
    assert stats.total_words == 4
    assert stats.entropy > 0.0


def test_vocab_inspector(mock_tokenizer):
    tokens = mock_tokenizer.encode("The quick brown fox")
    inspector = VocabularyInspector(mock_tokenizer)
    report = inspector.inspect(tokens)
    assert report.is_valid is True
    assert report.total_tokens == len(tokens)
    assert report.out_of_bounds_count == 0


# 5. SequenceBuilder Tests
def test_sequence_builder():
    token_ids = list(range(20))
    builder = SequenceBuilder(window_size=5, stride=5, drop_last=True)
    x_t, y_t = builder.build_sequences(token_ids)

    # With 20 tokens, required per sequence is L+1 = 6. 20 // 5 = 3 full sequences
    assert x_t.size(0) == 3
    assert y_t.size(0) == 3
    assert x_t.size(1) == 5
    assert y_t.size(1) == 5

    # Check next token alignment: Y[b, i] == X[b, i+1] except target shifts by 1
    # X[0] = [0, 1, 2, 3, 4], Y[0] = [1, 2, 3, 4, 5]
    assert torch.equal(x_t[0], torch.tensor([0, 1, 2, 3, 4]))
    assert torch.equal(y_t[0], torch.tensor([1, 2, 3, 4, 5]))


# 6. DatasetCache Tests
def test_dataset_cache(tmp_path):
    cache = DatasetCache(cache_dir=tmp_path / "cache", enabled=True)
    config_dict = {"window_size": 10, "stride": 10}

    x = torch.randint(0, 100, (5, 10))
    y = torch.randint(0, 100, (5, 10))

    cache.save("test_src", config_dict, x, y)
    loaded = cache.get("test_src", config_dict)
    assert loaded is not None
    loaded_x, loaded_y = loaded
    assert torch.equal(x, loaded_x)
    assert torch.equal(y, loaded_y)

    # Invalidation on modified config
    diff_config = {"window_size": 20, "stride": 10}
    assert cache.get("test_src", diff_config) is None


# 7. AuraTextDataset & AuraStreamingDataset Tests
def test_aura_text_dataset():
    x = torch.randint(0, 100, (10, 8))
    y = torch.randint(0, 100, (10, 8))
    ds = AuraTextDataset(x_tensor=x, y_tensor=y)

    assert len(ds) == 10
    sample_x, sample_y = ds[2]
    assert torch.equal(sample_x, x[2])
    assert torch.equal(sample_y, y[2])


def test_aura_streaming_dataset(tmp_path, mock_tokenizer):
    f = tmp_path / "stream_ds.txt"
    f.write_text("The quick brown fox jumps over the lazy dog.", encoding="utf-8")
    s_reader = StreamingReader([f])

    ds = AuraStreamingDataset(s_reader, mock_tokenizer, window_size=5, stride=5)
    samples = list(ds)
    assert len(samples) > 0
    x0, y0 = samples[0]
    assert x0.shape == (5,)
    assert y0.shape == (5,)


# 8. Splitter Tests
def test_dataset_splitter():
    x = torch.randint(0, 100, (100, 10))
    y = torch.randint(0, 100, (100, 10))
    ds = AuraTextDataset(x_tensor=x, y_tensor=y)

    splitter = DatasetSplitter(
        SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    )
    train_ds, val_ds, test_ds = splitter.split_dataset(ds)

    assert len(train_ds) == 80
    assert len(val_ds) == 10
    assert len(test_ds) == 10


# 9. Collate and BatchBuilder Tests
def test_collate_function():
    collate = CollateFunction(pad_token_id=0)
    s1_x = torch.tensor([1, 2, 3])
    s1_y = torch.tensor([2, 3, 4])
    s2_x = torch.tensor([5, 6, 7, 8, 9])
    s2_y = torch.tensor([6, 7, 8, 9, 10])

    batch = [(s1_x, s1_y), (s2_x, s2_y)]
    x_batch, y_batch = collate(batch)

    assert x_batch.shape == (2, 5)
    assert y_batch.shape == (2, 5)
    # Check padding in sample 1
    assert torch.equal(x_batch[0], torch.tensor([1, 2, 3, 0, 0]))


def test_batch_builder():
    x = torch.randint(0, 100, (32, 16))
    y = torch.randint(0, 100, (32, 16))
    ds = AuraTextDataset(x_tensor=x, y_tensor=y)

    builder = BatchBuilder(BatchConfig(batch_size=8, shuffle=False))
    loader = builder.build_dataloader(ds)

    batches = list(loader)
    assert len(batches) == 4
    x_b, y_b = batches[0]
    assert x_b.shape == (8, 16)
    assert y_b.shape == (8, 16)


# 10. End-to-End DatasetFactory Test
def test_dataset_factory(sample_text_file, mock_tokenizer):
    cfg = DatasetConfig(
        sequence=SequenceConfig(window_size=8, stride=8),
        cache=CacheConfig(enabled=False),
        split=SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42),
    )

    train_ds, val_ds, test_ds = DatasetFactory.build_pipeline(
        sample_text_file, mock_tokenizer, config=cfg
    )

    assert len(train_ds) > 0
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total > 0
