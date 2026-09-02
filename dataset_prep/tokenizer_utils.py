"""Shared tokenizer for token counting/chunking across the dataset-prep pipeline."""

_tokenizer = None


def get_tokenizer():
    """Lazily load the 'gpt2' tokenizer (tokenizer only — not a model class).

    Raises model_max_length past GPT-2's default 1024 so encoding a full
    corpus here doesn't log the misleading "...will result in indexing
    errors" warning — this tokenizer is only ever used to produce a flat
    token stream that chunk_text()/TokenBlockDataset later slice down to
    TechcodeXConfig.block_size before anything reaches the model.
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import GPT2TokenizerFast

        _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _tokenizer.model_max_length = int(1e30)
    return _tokenizer


def estimate_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return len(get_tokenizer().encode(text))
