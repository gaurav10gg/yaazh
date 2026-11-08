"""
Tests for src/text_normalizer.py.   Run with:  pytest -v

Every model we train depends on this normalizer, so these tests pin down its
behaviour: if a library update or an edit changes the romanization, a test fails
instead of the model silently learning a different spelling.
"""

import re

import pytest

from src.text_normalizer import (
    expand_numbers,
    find_unsupported_chars,
    normalize_text,
    number_to_tamil_words,
)

ONLY_LATIN_LETTERS = re.compile(r"[a-z]+")


# ---------------------------------------------------------------------------
# Numbers -> Tamil words
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("number, words", [
    (0, "பூஜ்ஜியம்"),
    (7, "ஏழு"),
    (10, "பத்து"),
    (11, "பதினொன்று"),
    (19, "பத்தொன்பது"),
    (20, "இருபது"),
    (24, "இருபத்து நான்கு"),
    (90, "தொண்ணூறு"),
    (99, "தொண்ணூற்று ஒன்பது"),
    (100, "நூறு"),
    (105, "நூற்று ஐந்து"),
    (900, "தொள்ளாயிரம்"),
    (999, "தொள்ளாயிரத்து தொண்ணூற்று ஒன்பது"),
    (1000, "ஆயிரம்"),
    (2024, "இரண்டாயிரத்து இருபத்து நான்கு"),
    (5000, "ஐயாயிரம்"),
    (9999, "ஒன்பதாயிரத்து தொள்ளாயிரத்து தொண்ணூற்று ஒன்பது"),
])
def test_number_to_tamil_words(number, words):
    assert number_to_tamil_words(number) == words


def test_every_number_up_to_9999_converts():
    for n in range(10_000):
        assert number_to_tamil_words(n)  # no KeyError, never empty


def test_large_numbers_are_read_digit_by_digit():
    assert number_to_tamil_words(12345) == "ஒன்று இரண்டு மூன்று நான்கு ஐந்து"


def test_expand_numbers_in_text():
    assert expand_numbers("3.5 கிலோ") == "மூன்று புள்ளி ஐந்து கிலோ"
    assert expand_numbers("1,000") == "ஆயிரம்"
    assert expand_numbers("007") == "பூஜ்ஜியம் பூஜ்ஜியம் ஏழு"


def test_tamil_digits_same_as_ascii_digits():
    assert normalize_text("௨௪") == normalize_text("24") == "irupaththu naanku"


def test_numbers_are_converted_before_romanizing():
    assert normalize_text("24 பேர்") == "irupaththu naanku peer"


# ---------------------------------------------------------------------------
# Vowels: long vowels doubled, short e/o without accents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tamil, latin", [
    ("அ", "a"), ("ஆ", "aa"),
    ("இ", "i"), ("ஈ", "ii"),
    ("உ", "u"), ("ஊ", "uu"),
    ("எ", "e"), ("ஏ", "ee"),
    ("ஒ", "o"), ("ஓ", "oo"),
    ("ஐ", "ai"), ("ஔ", "au"),
    ("கா", "kaa"), ("கீ", "kii"), ("கூ", "kuu"),
    ("கெ", "ke"), ("கே", "kee"), ("கொ", "ko"), ("கோ", "koo"),
])
def test_vowels(tamil, latin):
    assert normalize_text(tamil) == latin


# ---------------------------------------------------------------------------
# Consonants and Tamil-specific sounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tamil, latin", [
    ("தமிழ்", "thamizh"),        # ழ -> zh
    ("வாழை", "vaazhai"),
    ("வாளை", "vaalhai"),         # ள -> lh
    ("வாலை", "vaalai"),          # ல -> l
    ("காற்று", "kaarrrru"),      # ற -> rr
    ("என்ன", "enna"),            # ன (library has no entry for it) -> n
    ("அவன்", "avan"),
    ("கண்ணன்", "kanhnhan"),      # ண -> nh
    ("இல்லை", "illai"),
    ("பட்டம்", "pattam"),        # ட -> t
    ("பத்து", "paththu"),        # த -> th
    ("அம்மா", "ammaa"),
    ("தங்கம்", "thangkam"),      # ங -> ng
    ("கொஞ்சம்", "konjcham"),     # ஞ -> nj, ச -> ch
    ("எஃகு", "ehku"),            # ஃ -> h
])
def test_consonants(tamil, latin):
    assert normalize_text(tamil) == latin


def test_similar_sounds_stay_distinct():
    # ழ / ள / ல and ற / ர must not collapse into the same spelling
    assert len({normalize_text(w) for w in ["வாழை", "வாளை", "வாலை"]}) == 3
    assert normalize_text("கரை") != normalize_text("கறை")


CONSONANTS = list("கஙசஞடணதநபமயரலவழளறனஜஷஸஹ") + ["க்ஷ", "ஶ"]
VOWEL_SIGNS = ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ", "்"]
VOWELS = list("அஆஇஈஉஊஎஏஐஒஓஔஃ")


def test_every_letter_romanizes_to_plain_lowercase_latin():
    """Every consonant x vowel-sign combination must produce only a-z."""
    for consonant in CONSONANTS:
        for sign in VOWEL_SIGNS:
            result = normalize_text(consonant + sign)
            assert ONLY_LATIN_LETTERS.fullmatch(result), f"{consonant + sign!r} -> {result!r}"
    for vowel in VOWELS:
        result = normalize_text(vowel)
        assert ONLY_LATIN_LETTERS.fullmatch(result), f"{vowel!r} -> {result!r}"


# ---------------------------------------------------------------------------
# Cleanup: punctuation, whitespace, English, invisible characters
# ---------------------------------------------------------------------------

def test_punctuation_removed_and_whitespace_collapsed():
    assert normalize_text("  வணக்கம்,   உலகம்!!  ") == "vanhakkam ulakam"


def test_english_words_kept_but_lowercased():
    assert normalize_text("Hello உலகம்") == "hello ulakam"
    assert normalize_text("don't") == "dont"


def test_zero_width_characters_removed():
    assert normalize_text("கா‌லை") == normalize_text("காலை")


def test_real_dataset_sentence():
    sentence = "அதற்குத் தகுந்தபடி, ஏதாவது கொஞ்சம் பேசி, வேஷம் போட்டால் போகிறது."
    result = normalize_text(sentence)
    assert re.fullmatch(r"[a-z ]+", result), result
    assert result.startswith("atharrkuth thakunthapati eethaavathu")


# ---------------------------------------------------------------------------
# Vocabulary check
# ---------------------------------------------------------------------------

class FakeTokenizer:
    def get_vocab(self):
        return {letter: i for i, letter in enumerate("abcdefghijklmnopqrstuvwxyz")} | {"▁": 99}


def test_find_unsupported_chars_reports_leftovers():
    leftovers = find_unsupported_chars(["hello world", "café ü"], FakeTokenizer())
    assert leftovers == {"é": 1, "ü": 1}


def test_normalized_output_is_in_real_speecht5_vocab():
    """Uses the real SpeechT5 tokenizer (a small download); skipped if offline."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.SpeechT5Tokenizer.from_pretrained("microsoft/speecht5_tts")
    except Exception as error:  # no internet / not cached
        pytest.skip(f"SpeechT5 tokenizer unavailable: {error}")
    text = normalize_text("தமிழ் ஒரு அழகான மொழி. 2024-ல் 24 பேர் வந்தனர்! Hello")
    assert find_unsupported_chars([text], tokenizer) == {}
