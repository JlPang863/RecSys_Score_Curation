"""
Extract shallow text features from messages for multi-feature fusion.
Features are designed to correlate with instruction-following quality.
"""

import re
import numpy as np
import pandas as pd
from tqdm import tqdm


def _count_syllables(word: str) -> int:
    """Rough syllable count for English words."""
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and not word.endswith('le'):
        count = max(count - 1, 1)
    return max(count, 1)


def _sentence_lengths(text: str) -> list[int]:
    """Split text into sentences and return word counts per sentence."""
    sents = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sents if s.strip()]
    return lengths if lengths else [0]


def extract_features_from_messages(messages: list[dict]) -> dict[str, float]:
    """Extract quality-correlated features from a conversation."""
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    asst_texts = [m["content"] for m in messages if m["role"] == "assistant"]

    user_text = " ".join(user_texts)
    asst_text = " ".join(asst_texts)

    # --- Basic length features ---
    user_len = len(user_text)
    asst_len = len(asst_text)
    total_len = user_len + asst_len
    len_ratio = asst_len / max(user_len, 1)

    user_words = user_text.split()
    asst_words = asst_text.split()
    user_word_count = len(user_words)
    asst_word_count = len(asst_words)

    # --- Structural features ---
    num_turns = len(messages)
    num_user_turns = len(user_texts)
    num_asst_turns = len(asst_texts)

    # Sentence count (rough)
    asst_sent_lens = _sentence_lengths(asst_text)
    user_sent_lens = _sentence_lengths(user_text)
    asst_sentences = len(asst_sent_lens)
    user_sentences = len(user_sent_lens)

    # Paragraph count
    asst_paras = [p for p in asst_text.split("\n\n") if p.strip()]
    asst_paragraphs = len(asst_paras)

    # Average word length (vocabulary sophistication proxy)
    word_lens_asst = [len(w) for w in asst_words]
    word_lens_user = [len(w) for w in user_words]
    avg_word_len_asst = np.mean(word_lens_asst) if word_lens_asst else 0.0
    avg_word_len_user = np.mean(word_lens_user) if word_lens_user else 0.0

    # Average sentence length
    avg_sent_len_asst = asst_word_count / max(asst_sentences, 1)

    # NEW: Sentence length statistics (response)
    max_sent_len_asst = max(asst_sent_lens) if asst_sent_lens else 0
    min_sent_len_asst = min(asst_sent_lens) if asst_sent_lens else 0
    std_sent_len_asst = float(np.std(asst_sent_lens)) if len(asst_sent_lens) > 1 else 0.0

    # NEW: Average paragraph length
    avg_para_len_asst = asst_word_count / max(asst_paragraphs, 1)

    # NEW: Max word length (vocabulary sophistication)
    max_word_len_asst = max(word_lens_asst) if word_lens_asst else 0

    # --- Vocabulary diversity ---
    asst_words_lower = [w.lower() for w in asst_words]
    user_words_lower = [w.lower() for w in user_words]
    unique_words_asst = len(set(asst_words_lower))
    vocab_diversity_asst = unique_words_asst / max(asst_word_count, 1)

    unique_words_user = len(set(user_words_lower))
    vocab_diversity_user = unique_words_user / max(user_word_count, 1)

    # NEW: Hapax legomena ratio (words appearing exactly once / total)
    if asst_words_lower:
        from collections import Counter
        word_freq = Counter(asst_words_lower)
        hapax_count = sum(1 for c in word_freq.values() if c == 1)
        hapax_ratio = hapax_count / max(asst_word_count, 1)
    else:
        hapax_ratio = 0.0

    # --- Readability features ---
    # Flesch-Kincaid approximation
    total_syllables = sum(_count_syllables(w) for w in asst_words) if asst_words else 0
    avg_syllables_per_word = total_syllables / max(asst_word_count, 1)
    # FK Grade Level ≈ 0.39*(words/sent) + 11.8*(syllables/word) - 15.59
    fk_grade = 0.39 * avg_sent_len_asst + 11.8 * avg_syllables_per_word - 15.59
    # Coleman-Liau Index approximation: 0.0588*L - 0.296*S - 15.8
    # L = avg letters per 100 words, S = avg sentences per 100 words
    L = (sum(len(w) for w in asst_words) / max(asst_word_count, 1)) * 100
    S = (asst_sentences / max(asst_word_count, 1)) * 100
    coleman_liau = 0.0588 * L - 0.296 * S - 15.8

    # --- Content quality signals ---
    # Code blocks
    code_block_pairs = len(re.findall(r'```', asst_text)) // 2
    has_code = 1.0 if code_block_pairs > 0 else 0.0
    # NEW: Inline code
    inline_code = len(re.findall(r'`[^`]+`', asst_text))

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

    # NEW: Markdown tables
    table_rows = len(re.findall(r'^\s*\|.*\|', asst_text, re.MULTILINE))
    has_table = 1.0 if table_rows > 2 else 0.0

    # NEW: Bold / italic text
    bold_count = len(re.findall(r'\*\*[^*]+\*\*', asst_text))
    italic_count = len(re.findall(r'(?<!\*)\*(?!\*)[^*]+\*(?!\*)', asst_text))

    # NEW: Math indicators
    math_indicators = len(re.findall(r'[\+\-\*/=<>≤≥≠∑∏∫]', asst_text))
    math_density = math_indicators / max(asst_len, 1)

    # NEW: Parenthetical expressions (often used for explanations)
    paren_count = len(re.findall(r'\([^)]+\)', asst_text))

    # NEW: Punctuation diversity (distinct punctuation types used)
    punct_types = set(re.findall(r'[^\w\s]', asst_text))
    punct_diversity = len(punct_types)

    # NEW: Digit density (numbers in response)
    digit_count = sum(c.isdigit() for c in asst_text)
    digit_density = digit_count / max(asst_len, 1)

    # --- Prompt complexity features ---
    question_marks = user_text.count("?")
    has_question = 1.0 if question_marks > 0 else 0.0

    instruction_keywords = sum(1 for kw in
        ["write", "explain", "describe", "list", "create", "generate",
         "translate", "summarize", "compare", "analyze", "help", "how",
         "what", "why", "can you", "please"]
        if kw in user_text.lower())

    # NEW: Prompt length category (short/medium/long)
    log_user_word_count = np.log1p(user_word_count)

    # NEW: Prompt has code
    user_has_code = 1.0 if '```' in user_text or '`' in user_text else 0.0

    # NEW: Prompt complexity - number of constraints/requirements
    constraint_words = sum(1 for kw in
        ["must", "should", "need", "require", "ensure", "make sure",
         "at least", "at most", "no more than", "exactly", "only",
         "don't", "do not", "avoid", "without", "except"]
        if kw in user_text.lower())

    # --- Prompt-Response alignment ---
    # NEW: Word overlap between prompt and response
    if user_words_lower and asst_words_lower:
        user_set = set(user_words_lower)
        asst_set = set(asst_words_lower)
        # Remove stopwords for more meaningful overlap
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'and', 'but', 'or', 'not', 'no',
                     'it', 'its', 'this', 'that', 'i', 'you', 'he', 'she', 'we',
                     'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your'}
        user_content = user_set - stopwords
        asst_content = asst_set - stopwords
        if user_content and asst_content:
            overlap = len(user_content & asst_content)
            word_overlap_ratio = overlap / max(len(user_content), 1)
        else:
            word_overlap_ratio = 0.0
    else:
        word_overlap_ratio = 0.0

    # --- Response quality heuristics ---
    starts_capital = 1.0 if asst_text and asst_text[0].isupper() else 0.0
    ends_punct = 1.0 if asst_text and asst_text.rstrip()[-1:] in ".!?\"')" else 0.0

    # Repetition: bigram
    if len(asst_words) >= 2:
        bigrams = list(zip(asst_words[:-1], asst_words[1:]))
        unique_bigrams = len(set(bigrams))
        bigram_repeat_ratio = 1.0 - unique_bigrams / len(bigrams)
    else:
        bigram_repeat_ratio = 0.0

    # NEW: Trigram repetition
    if len(asst_words) >= 3:
        trigrams = list(zip(asst_words[:-2], asst_words[1:-1], asst_words[2:]))
        unique_trigrams = len(set(trigrams))
        trigram_repeat_ratio = 1.0 - unique_trigrams / len(trigrams)
    else:
        trigram_repeat_ratio = 0.0

    # "I" usage
    i_count = sum(1 for w in asst_words if w.lower() == "i")
    i_density = i_count / max(asst_word_count, 1)

    # NEW: Politeness / compliance indicators
    asst_lower = asst_text.lower()
    compliance_phrases = sum(1 for phrase in
        ["sure", "certainly", "of course", "here is", "here are",
         "here's", "let me", "i'll", "i will", "happy to"]
        if phrase in asst_lower)

    # NEW: Refusal indicators
    refusal_phrases = sum(1 for phrase in
        ["i cannot", "i can't", "i'm sorry", "i am sorry",
         "i'm unable", "i am unable", "unfortunately",
         "i don't think", "i'm not able", "as an ai"]
        if phrase in asst_lower)

    # NEW: Hedging language
    hedge_words = sum(1 for w in asst_words_lower
        if w in {"maybe", "perhaps", "possibly", "might", "somewhat",
                 "generally", "typically", "usually", "approximately"})
    hedge_density = hedge_words / max(asst_word_count, 1)

    # NEW: Transition words (coherence signal)
    transition_words = sum(1 for w in asst_words_lower
        if w in {"however", "therefore", "furthermore", "moreover",
                 "additionally", "consequently", "nevertheless",
                 "meanwhile", "alternatively", "specifically",
                 "similarly", "conversely", "accordingly"})
    transition_density = transition_words / max(asst_word_count, 1)

    # NEW: Example/evidence phrases
    example_phrases = sum(1 for phrase in
        ["for example", "for instance", "such as", "e.g.", "i.e.",
         "consider", "specifically", "in particular", "namely"]
        if phrase in asst_lower)

    return {
        # Length (8)
        "user_char_len": user_len,
        "asst_char_len": asst_len,
        "total_char_len": total_len,
        "len_ratio": len_ratio,
        "user_word_count": user_word_count,
        "asst_word_count": asst_word_count,
        "log_asst_len": np.log1p(asst_len),
        "log_user_len": np.log1p(user_len),
        # Structure (12)
        "num_turns": num_turns,
        "num_user_turns": num_user_turns,
        "num_asst_turns": num_asst_turns,
        "asst_sentences": asst_sentences,
        "user_sentences": user_sentences,
        "asst_paragraphs": asst_paragraphs,
        "avg_word_len_asst": avg_word_len_asst,
        "avg_word_len_user": avg_word_len_user,
        "avg_sent_len_asst": avg_sent_len_asst,
        "max_sent_len_asst": max_sent_len_asst,
        "min_sent_len_asst": min_sent_len_asst,
        "std_sent_len_asst": std_sent_len_asst,
        # Paragraph (2)
        "avg_para_len_asst": avg_para_len_asst,
        "max_word_len_asst": max_word_len_asst,
        # Vocabulary (3)
        "vocab_diversity_asst": vocab_diversity_asst,
        "vocab_diversity_user": vocab_diversity_user,
        "hapax_ratio": hapax_ratio,
        # Readability (3)
        "avg_syllables_per_word": avg_syllables_per_word,
        "fk_grade": fk_grade,
        "coleman_liau": coleman_liau,
        # Content signals (16)
        "has_code": has_code,
        "code_block_pairs": code_block_pairs,
        "inline_code": inline_code,
        "has_list": has_list,
        "list_items": list_items,
        "header_count": header_count,
        "url_count": url_count,
        "special_density": special_density,
        "newline_density": newline_density,
        "newline_count": newline_count,
        "has_table": has_table,
        "table_rows": table_rows,
        "bold_count": bold_count,
        "italic_count": italic_count,
        "math_density": math_density,
        "paren_count": paren_count,
        # Text statistics (3)
        "punct_diversity": punct_diversity,
        "digit_density": digit_density,
        "log_user_word_count": log_user_word_count,
        # Prompt (5)
        "question_marks": question_marks,
        "has_question": has_question,
        "instruction_keywords": instruction_keywords,
        "user_has_code": user_has_code,
        "constraint_words": constraint_words,
        # Alignment (1)
        "word_overlap_ratio": word_overlap_ratio,
        # Response quality (10)
        "starts_capital": starts_capital,
        "ends_punct": ends_punct,
        "bigram_repeat_ratio": bigram_repeat_ratio,
        "trigram_repeat_ratio": trigram_repeat_ratio,
        "i_density": i_density,
        "compliance_phrases": compliance_phrases,
        "refusal_phrases": refusal_phrases,
        "hedge_density": hedge_density,
        "transition_density": transition_density,
        "example_phrases": example_phrases,
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
