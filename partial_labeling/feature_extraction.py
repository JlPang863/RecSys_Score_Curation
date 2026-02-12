"""
Extract shallow text features from messages for multi-feature fusion.
Features are designed to correlate with instruction-following quality.
"""

import re
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_features_from_messages(messages: list[dict]) -> dict[str, float]:
    """Extract quality-correlated features from a conversation."""
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    asst_texts = [m["content"] for m in messages if m["role"] == "assistant"]

    user_text = " ".join(user_texts)
    asst_text = " ".join(asst_texts)
    full_text = user_text + " " + asst_text

    # --- Basic length features ---
    user_len = len(user_text)
    asst_len = len(asst_text)
    total_len = user_len + asst_len
    len_ratio = asst_len / max(user_len, 1)  # response/prompt length ratio

    user_words = user_text.split()
    asst_words = asst_text.split()
    user_word_count = len(user_words)
    asst_word_count = len(asst_words)

    # --- Structural features ---
    num_turns = len(messages)
    num_user_turns = len(user_texts)
    num_asst_turns = len(asst_texts)

    # Sentence count (rough)
    asst_sentences = len(re.split(r'[.!?]+', asst_text))
    user_sentences = len(re.split(r'[.!?]+', user_text))

    # Paragraph count
    asst_paragraphs = len([p for p in asst_text.split("\n\n") if p.strip()])

    # Average word length (vocabulary sophistication proxy)
    avg_word_len_asst = np.mean([len(w) for w in asst_words]) if asst_words else 0.0
    avg_word_len_user = np.mean([len(w) for w in user_words]) if user_words else 0.0

    # Average sentence length
    avg_sent_len_asst = asst_word_count / max(asst_sentences, 1)

    # --- Vocabulary diversity ---
    unique_words_asst = len(set(w.lower() for w in asst_words))
    vocab_diversity_asst = unique_words_asst / max(asst_word_count, 1)

    unique_words_user = len(set(w.lower() for w in user_words))
    vocab_diversity_user = unique_words_user / max(user_word_count, 1)

    # --- Content quality signals ---
    # Code blocks
    code_blocks = len(re.findall(r'```', asst_text))
    has_code = 1.0 if code_blocks > 0 else 0.0

    # Bullet points / numbered lists
    bullet_count = len(re.findall(r'^\s*[-*•]\s', asst_text, re.MULTILINE))
    numbered_count = len(re.findall(r'^\s*\d+[.)]\s', asst_text, re.MULTILINE))
    has_list = 1.0 if (bullet_count + numbered_count) > 0 else 0.0
    list_items = bullet_count + numbered_count

    # Headers (markdown)
    header_count = len(re.findall(r'^#+\s', asst_text, re.MULTILINE))

    # URLs
    url_count = len(re.findall(r'https?://\S+', asst_text))

    # Special characters density
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', asst_text))
    special_density = special_chars / max(asst_len, 1)

    # Newline density (formatting effort)
    newline_count = asst_text.count("\n")
    newline_density = newline_count / max(asst_len, 1)

    # --- Prompt complexity features ---
    # Question marks in prompt
    question_marks = user_text.count("?")
    has_question = 1.0 if question_marks > 0 else 0.0

    # Instruction keywords
    instruction_keywords = sum(1 for kw in
        ["write", "explain", "describe", "list", "create", "generate",
         "translate", "summarize", "compare", "analyze", "help", "how",
         "what", "why", "can you", "please"]
        if kw in user_text.lower())

    # --- Response quality heuristics ---
    # Starts with capital letter
    starts_capital = 1.0 if asst_text and asst_text[0].isupper() else 0.0

    # Ends with proper punctuation
    ends_punct = 1.0 if asst_text and asst_text.rstrip()[-1:] in ".!?\"')" else 0.0

    # Repetition: fraction of repeated bigrams
    if len(asst_words) >= 2:
        bigrams = list(zip(asst_words[:-1], asst_words[1:]))
        unique_bigrams = len(set(bigrams))
        bigram_repeat_ratio = 1.0 - unique_bigrams / len(bigrams)
    else:
        bigram_repeat_ratio = 0.0

    # "I" usage (first person, common in good explanations)
    i_count = sum(1 for w in asst_words if w.lower() == "i")
    i_density = i_count / max(asst_word_count, 1)

    return {
        # Length
        "user_char_len": user_len,
        "asst_char_len": asst_len,
        "total_char_len": total_len,
        "len_ratio": len_ratio,
        "user_word_count": user_word_count,
        "asst_word_count": asst_word_count,
        "log_asst_len": np.log1p(asst_len),
        "log_user_len": np.log1p(user_len),
        # Structure
        "num_turns": num_turns,
        "num_user_turns": num_user_turns,
        "num_asst_turns": num_asst_turns,
        "asst_sentences": asst_sentences,
        "user_sentences": user_sentences,
        "asst_paragraphs": asst_paragraphs,
        "avg_word_len_asst": avg_word_len_asst,
        "avg_word_len_user": avg_word_len_user,
        "avg_sent_len_asst": avg_sent_len_asst,
        # Vocabulary
        "vocab_diversity_asst": vocab_diversity_asst,
        "vocab_diversity_user": vocab_diversity_user,
        # Content signals
        "has_code": has_code,
        "code_blocks": code_blocks,
        "has_list": has_list,
        "list_items": list_items,
        "header_count": header_count,
        "url_count": url_count,
        "special_density": special_density,
        "newline_density": newline_density,
        "newline_count": newline_count,
        # Prompt
        "question_marks": question_marks,
        "has_question": has_question,
        "instruction_keywords": instruction_keywords,
        # Response quality
        "starts_capital": starts_capital,
        "ends_punct": ends_punct,
        "bigram_repeat_ratio": bigram_repeat_ratio,
        "i_density": i_density,
    }


def extract_features_batch(
    df: pd.DataFrame,
    messages_key: str = "messages",
    show_progress: bool = True,
) -> np.ndarray:
    """Extract features for all rows, return (N, F) float32 array."""
    feature_dicts = []
    for msgs in tqdm(
        df[messages_key].tolist(),
        desc="extract features",
        disable=not show_progress,
    ):
        feature_dicts.append(extract_features_from_messages(msgs))

    feature_names = list(feature_dicts[0].keys())
    features = np.array(
        [[d[name] for name in feature_names] for d in feature_dicts],
        dtype=np.float32,
    )
    return features, feature_names


def normalize_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score normalize using train statistics."""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.maximum(std, 1e-8)  # avoid division by zero
    return (train_features - mean) / std, (test_features - mean) / std
