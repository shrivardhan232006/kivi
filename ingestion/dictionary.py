"""Dictionary correction layer — deterministic lookup-and-replace."""

import re
from db.database import fetchall, execute, commit


def load_dictionary() -> list[dict]:
    """Load all dictionary entries."""
    return fetchall("SELECT raw_form, corrected_form FROM dictionary ORDER BY length(raw_form) DESC")


def _deduplicate_consecutive_tokens(text: str) -> str:
    """Remove consecutively repeated tokens to fix double-name artifacts.

    e.g. 'Priya Priya Reddy' → 'Priya Reddy',
         'Alex Alex Rivera'  → 'Alex Rivera'.
    """
    words = text.split()
    if len(words) <= 1:
        return text
    result = [words[0]]
    for w in words[1:]:
        if w.lower() != result[-1].lower():
            result.append(w)
    return ' '.join(result)


def apply_corrections(text: str, dictionary: list[dict] | None = None) -> str:
    """Apply dictionary corrections to text. Longest-match-first, whole-word, case-insensitive.

    After replacement, consecutive duplicate tokens are removed to prevent
    artifacts like 'Priya Priya Reddy'.
    """
    if dictionary is None:
        dictionary = load_dictionary()
    if not dictionary:
        return text
    for entry in dictionary:
        # Whole-word boundary matching, case-insensitive
        pattern = re.compile(r'\b' + re.escape(entry['raw_form']) + r'\b', re.IGNORECASE)
        text = pattern.sub(entry['corrected_form'], text)
    # Clean up double-name artifacts
    text = _deduplicate_consecutive_tokens(text)
    return text


def add_entry(raw_form: str, corrected_form: str):
    """Add a dictionary entry."""
    execute(
        "INSERT OR REPLACE INTO dictionary (raw_form, corrected_form, occurrence_count) VALUES (?, ?, 1)",
        (raw_form.lower(), corrected_form)
    )
    commit()


def delete_entry(raw_form: str):
    """Delete a dictionary entry."""
    execute("DELETE FROM dictionary WHERE raw_form = ?", (raw_form.lower(),))
    commit()


def track_correction(raw_form: str, corrected_form: str):
    """Track a raw→corrected pair for auto-suggestion."""
    from config import DICTIONARY_SUGGESTION_THRESHOLD

    raw_lower = raw_form.lower()
    corrected = corrected_form.strip()

    # Skip if already in dictionary
    existing = fetchall("SELECT 1 FROM dictionary WHERE raw_form = ?", (raw_lower,))
    if existing:
        return

    # Check if suggestion already exists
    suggestion = fetchall(
        "SELECT id, occurrence_count FROM dictionary_suggestions WHERE raw_form = ? AND corrected_form = ? AND status = 'pending'",
        (raw_lower, corrected)
    )
    if suggestion:
        new_count = suggestion[0]['occurrence_count'] + 1
        execute(
            "UPDATE dictionary_suggestions SET occurrence_count = ? WHERE id = ?",
            (new_count, suggestion[0]['id'])
        )
    else:
        execute(
            "INSERT INTO dictionary_suggestions (raw_form, corrected_form, occurrence_count) VALUES (?, ?, 1)",
            (raw_lower, corrected)
        )
    commit()


def auto_populate_from_corpus(records: list[dict]):
    """Populate dictionary from corpus name_correction patterns where raw→formatted corrections are consistent."""
    import difflib

    correction_counts: dict[tuple[str, str], int] = {}

    for rec in records:
        raw = rec.get('raw_asr_output', '')
        formatted = rec.get('llm_formatted_output', '')
        entities = rec.get('entities', [])

        if not entities:
            continue

        # For each entity mentioned, check if the raw text has a misspelling
        raw_lower = raw.lower()
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in raw_lower:
                continue  # Already correct in raw

            # Find close matches in the raw text (words that look like the entity)
            raw_words = raw_lower.split()
            entity_words = entity_lower.split()

            if len(entity_words) == 0:
                continue

            # Sliding window over raw words
            for i in range(len(raw_words) - len(entity_words) + 1):
                candidate = ' '.join(raw_words[i:i + len(entity_words)])
                ratio = difflib.SequenceMatcher(None, candidate, entity_lower).ratio()
                if 0.6 < ratio < 1.0:  # Close but not exact
                    key = (candidate, entity)
                    correction_counts[key] = correction_counts.get(key, 0) + 1

    # Add corrections that appear consistently (2+ times for auto-populate)
    added = 0
    for (raw_form, corrected_form), count in correction_counts.items():
        if count >= 2:
            existing = fetchall("SELECT 1 FROM dictionary WHERE raw_form = ?", (raw_form,))
            if not existing:
                execute(
                    "INSERT OR IGNORE INTO dictionary (raw_form, corrected_form, occurrence_count) VALUES (?, ?, ?)",
                    (raw_form, corrected_form, count)
                )
                added += 1

    commit()
    return added
