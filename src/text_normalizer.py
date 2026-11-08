"""
Tamil text normalizer for Yaazh.

WHY THIS EXISTS
---------------
SpeechT5's tokenizer was trained on English text, so it only knows Latin
letters. If we fed it Tamil script, every character would become <unk>
("unknown") and the model could learn nothing. So we turn Tamil into a
consistent Latin spelling ("romanization"):

    "தமிழ் 24"  ->  "thamizh irupaththu naanku"

THE PIPELINE (normalize_text runs these in order)
-------------------------------------------------
  1. Clean up Unicode (invisible zero-width characters, Tamil digits ௧௨௩ -> 123).
  2. Numbers -> Tamil words      24 -> இருபத்து நான்கு   (BEFORE romanizing,
     so numbers are spoken in Tamil, not English).
  3. Romanize each run of Tamil letters (English words are left alone):
       a. ன -> ந  (the library has no entry for ன, see below)
       b. indic-transliteration: Tamil -> ITRANS
       c. one pass over the ITRANS output, rewriting each token to our spelling
          (long vowels doubled, Tamil sounds -> one fixed spelling each)
  4. Lowercase.
  5. Character replacement map (fixes any leftover unsupported characters).
  6. Remove punctuation, collapse whitespace.

THE SAME FUNCTION IS USED EVERYWHERE
------------------------------------
prepare_data.py, inference.py and app.py all call normalize_text(). If training
and inference romanized text differently, the model would hear "new" spellings
at inference time and mispronounce them.
"""

import re
import sys
import unicodedata
from collections import Counter

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ===========================================================================
# Step 1: Unicode cleanup
# ===========================================================================

# Zero-width characters are invisible but real; they'd become <unk> tokens.
ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")

# Tamil has its own digit symbols. Convert them to 0-9 so step 2 handles both.
TAMIL_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")

# ===========================================================================
# Step 2: Numbers -> Tamil words (0-9999; larger numbers are read digit by digit)
# ===========================================================================

UNITS = ["பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு",
         "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"]

TEENS = ["பத்து", "பதினொன்று", "பன்னிரண்டு", "பதின்மூன்று", "பதினான்கு",
         "பதினைந்து", "பதினாறு", "பதினேழு", "பதினெட்டு", "பத்தொன்பது"]  # 10-19

# Tamil uses a different form when more words follow:
#   20 = இருபது, but 24 = இருபத்து நான்கு
TENS = {2: "இருபது", 3: "முப்பது", 4: "நாற்பது", 5: "ஐம்பது",
        6: "அறுபது", 7: "எழுபது", 8: "எண்பது", 9: "தொண்ணூறு"}
TENS_JOINING = {2: "இருபத்து", 3: "முப்பத்து", 4: "நாற்பத்து", 5: "ஐம்பத்து",
                6: "அறுபத்து", 7: "எழுபத்து", 8: "எண்பத்து", 9: "தொண்ணூற்று"}

HUNDREDS = {1: "நூறு", 2: "இருநூறு", 3: "முந்நூறு", 4: "நானூறு", 5: "ஐந்நூறு",
            6: "அறுநூறு", 7: "எழுநூறு", 8: "எண்ணூறு", 9: "தொள்ளாயிரம்"}
HUNDREDS_JOINING = {1: "நூற்று", 2: "இருநூற்று", 3: "முந்நூற்று", 4: "நானூற்று",
                    5: "ஐந்நூற்று", 6: "அறுநூற்று", 7: "எழுநூற்று", 8: "எண்ணூற்று",
                    9: "தொள்ளாயிரத்து"}

THOUSANDS = {1: "ஆயிரம்", 2: "இரண்டாயிரம்", 3: "மூவாயிரம்", 4: "நான்காயிரம்",
             5: "ஐயாயிரம்", 6: "ஆறாயிரம்", 7: "ஏழாயிரம்", 8: "எட்டாயிரம்",
             9: "ஒன்பதாயிரம்"}
THOUSANDS_JOINING = {1: "ஆயிரத்து", 2: "இரண்டாயிரத்து", 3: "மூவாயிரத்து",
                     4: "நான்காயிரத்து", 5: "ஐயாயிரத்து", 6: "ஆறாயிரத்து",
                     7: "ஏழாயிரத்து", 8: "எட்டாயிரத்து", 9: "ஒன்பதாயிரத்து"}

DECIMAL_POINT = "புள்ளி"  # "point", as in 3.5 = மூன்று புள்ளி ஐந்து


def _below_100(n: int) -> str:
    """1-99 in Tamil words."""
    if n < 10:
        return UNITS[n]
    if n < 20:
        return TEENS[n - 10]
    tens, unit = divmod(n, 10)
    if unit == 0:
        return TENS[tens]
    return f"{TENS_JOINING[tens]} {UNITS[unit]}"


def _digit_by_digit(digits: str) -> str:
    return " ".join(UNITS[int(d)] for d in digits)


def number_to_tamil_words(n: int) -> str:
    """
    Convert an integer to Tamil words.
        24   -> இருபத்து நான்கு
        2024 -> இரண்டாயிரத்து இருபத்து நான்கு
    Numbers above 9999 are read digit by digit (like phone numbers).
    """
    if n < 0:
        raise ValueError("negative numbers are not supported")
    if n == 0:
        return UNITS[0]
    if n > 9999:
        return _digit_by_digit(str(n))

    words = []
    thousands, rest = divmod(n, 1000)
    if thousands:
        words.append(THOUSANDS_JOINING[thousands] if rest else THOUSANDS[thousands])
    hundreds, rest = divmod(rest, 100)
    if hundreds:
        words.append(HUNDREDS_JOINING[hundreds] if rest else HUNDREDS[hundreds])
    if rest:
        words.append(_below_100(rest))
    return " ".join(words)


def _number_string_to_tamil(number: str) -> str:
    """Handle one matched number string such as '24', '007' or '3.14'."""
    if "." in number:
        whole, fraction = number.split(".")
        return f"{_number_string_to_tamil(whole)} {DECIMAL_POINT} {_digit_by_digit(fraction)}"
    if len(number) > 1 and number.startswith("0"):
        return _digit_by_digit(number)  # codes like 007 are read digit by digit
    return number_to_tamil_words(int(number))


THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d)")  # the comma in 1,000
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def expand_numbers(text: str) -> str:
    text = THOUSANDS_SEPARATOR.sub("", text)
    return NUMBER.sub(lambda m: _number_string_to_tamil(m.group(0)), text)


# ===========================================================================
# Step 3: Romanization (Tamil script -> Latin letters)
# ===========================================================================

# Any run of characters from the Tamil Unicode block (U+0B80-U+0BFF).
TAMIL_RUN = re.compile(r"[஀-௿]+")

# 3a. Fix BEFORE transliterating.
# indic-transliteration's Tamil table was designed for writing *Sanskrit* in
# Tamil script, and Sanskrit has no "ன". So ன is not in the table and would
# pass through untransliterated (and its ் marker would be lost). ன and ந are
# both the "n" sound, so we write ன as ந first.
PRE_TRANSLITERATION = {"ன": "ந"}

# 3c. Rewrite the library's ITRANS tokens into our final spelling.
#
# Because the library maps Sanskrit -> Tamil many-to-one (क ख ग घ all -> க),
# going backwards it outputs the LAST match, e.g. க -> "gh" (घ). These outputs
# are odd but completely predictable, so we translate each one exactly once.
#
#   Tamil   library output   our spelling
ITRANS_TO_YAAZH = {
    # --- long vowels: uppercase -> doubled lowercase (keeps length info) ---
    "A": "aa",    # ஆ  ா
    "I": "ii",    # ஈ  ீ
    "U": "uu",    # ஊ  ூ
    "e": "ee",    # ஏ  ே   (the library writes long e as plain "e")
    "o": "oo",    # ஓ  ோ   (and long o as plain "o")
    # --- short e / o come out with accents, which SpeechT5 doesn't know ---
    "è": "e",     # எ  ெ
    "ò": "o",     # ஒ  ொ
    # --- consonants the library writes as Sanskrit "voiced, aspirated" ---
    "gh": "k",    # க
    "jh": "ch",   # ச
    "Dh": "t",    # ட  (retroflex t)
    "dh": "th",   # த  (dental t, as in Tamil-English "thamizh")
    "bh": "p",    # ப
    # --- nasals ---
    "~N": "ng",   # ங
    "~n": "nj",   # ஞ
    "N": "nh",    # ண  (retroflex n, kept distinct from ந/ன)
    "M": "m",     # ம் (the library writes a final ம் as Sanskrit "M")
    # --- Tamil-specific sounds: one fixed spelling each ---
    "L": "lh",    # ள  (retroflex l, distinct from ல = "l" and ழ = "zh")
    "R": "rr",    # ற  (distinct from ர = "r")
    # ழ is already "zh" in the library output - kept as is.
    # --- other letters ---
    "H": "h",     # ஃ  (aytham)
    "Sh": "sh",   # ஷ
    "kSh": "ksh", # க்ஷ
    "OM": "om",   # ௐ
}

# Longest tokens first, so "kSh" wins over "Sh", and "Dh" over a lone "D".
_ITRANS_TOKEN = re.compile(
    "|".join(re.escape(token) for token in sorted(ITRANS_TO_YAAZH, key=len, reverse=True))
)


def romanize_tamil(tamil: str) -> str:
    """Romanize a string that contains only Tamil characters."""
    for source, target in PRE_TRANSLITERATION.items():
        tamil = tamil.replace(source, target)
    itrans = transliterate(tamil, sanscript.TAMIL, sanscript.ITRANS)
    # One single pass: each token is replaced once, so "e" -> "ee" never
    # touches the "e" we just produced from "è".
    return _ITRANS_TOKEN.sub(lambda m: ITRANS_TO_YAAZH[m.group(0)], itrans)


# ===========================================================================
# Steps 5-6: leftovers, punctuation, whitespace
# ===========================================================================

# Characters found by the vocabulary check (find_unsupported_chars) and how to fix them.
CHAR_REPLACEMENTS = {
    "'": "",   # don't -> dont (instead of "don t")
    "’": "",
    "‘": "",
}

# Anything that is not a letter/digit/space (in ANY script) is punctuation or
# a symbol -> becomes a space. Leftover letters from other scripts are NOT
# removed here on purpose, so find_unsupported_chars() can report them.
PUNCTUATION = re.compile(r"[^\w\s]|_")
WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Turn raw Tamil text into the lowercase Latin text SpeechT5 is trained on.
    Used by training, inference and the demo - always this exact function.
    """
    text = unicodedata.normalize("NFC", text)          # 1. one canonical form for each letter
    text = ZERO_WIDTH.sub("", text)
    text = text.translate(TAMIL_DIGITS)
    text = expand_numbers(text)                         # 2. 24 -> இருபத்து நான்கு
    text = TAMIL_RUN.sub(lambda m: romanize_tamil(m.group(0)), text)  # 3. Tamil -> Latin
    text = text.lower()                                 # 4. English words get lowercased too
    for source, target in CHAR_REPLACEMENTS.items():    # 5. fix known leftovers
        text = text.replace(source, target)
    text = PUNCTUATION.sub(" ", text)                   # 6. remove punctuation
    return WHITESPACE.sub(" ", text).strip()


# ===========================================================================
# Vocabulary check: is every character something SpeechT5 knows?
# ===========================================================================

def supported_characters(tokenizer) -> set:
    """Single characters in the SpeechT5 vocabulary (plus space, written '▁' there)."""
    return {token for token in tokenizer.get_vocab() if len(token) == 1} | {" "}


def find_unsupported_chars(normalized_texts, tokenizer) -> Counter:
    """Count characters in already-normalized texts that SpeechT5 cannot tokenize."""
    supported = supported_characters(tokenizer)
    return Counter(ch for text in normalized_texts for ch in text if ch not in supported)


if __name__ == "__main__":
    # Try it:  python -m src.text_normalizer "தமிழ் 24"
    sys.stdout.reconfigure(encoding="utf-8")
    sample = " ".join(sys.argv[1:]) or "வணக்கம்! இன்று 24 பேர் வந்தனர். தமிழ் ஒரு அழகான மொழி."
    print("input     :", sample)
    print("normalized:", normalize_text(sample))
