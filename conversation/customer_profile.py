"""
customer_profile.py

The structured record of everything known about the caller, plus the
rule-based parsing that turns their spoken (transcribed) answers into field
values.

Extraction is deliberately simple regex/keyword matching rather than an NLU
model or an LLM call - the brief excludes tool calling, function calling,
and any extra LLM round-trips, so this keeps the module self-contained,
dependency-free, and easy to unit test. It won't catch every phrasing a
caller might use; when it can't confidently parse an answer, it returns
False/None and leaves the field empty, so the Conversation Manager will
naturally ask the same question again next turn. Swap `apply_raw_answer`'s
parsers for a smarter extractor later without touching any other file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Set
from .amounts import format_indian_currency_for_speech

# --- lightweight vocab used by the parsers below ----------------------------

_WORD_RE = re.compile(r"[a-z']+")

_NUMBER_WORD_PATTERN = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|and"
)
_AMOUNT_UNIT_ALIAS_RE = re.compile(
    rf"\b(?P<amount>(?:\d+(?:\.\d+)?|(?:{_NUMBER_WORD_PATTERN})(?:[\s-]+(?:{_NUMBER_WORD_PATTERN})){{0,8}}))"
    r"\s+(?:black|lac|lacs)\s*(?P<marker>rupees?|rs\.?|inr|cover(?:age)?|premium|amount|budget)\b",
    flags=re.IGNORECASE,
)
_STT_PHRASE_ALIASES = (
    (re.compile(r"\bcash\s+less\b", flags=re.IGNORECASE), "cashless"),
    (re.compile(r"\bco\s*-?\s*pay(?:ment)?\b", flags=re.IGNORECASE), "copay"),
    (re.compile(r"\bsum\s+(?:in|and)\s+short\b", flags=re.IGNORECASE), "sum insured"),
)

_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "haan", "ha", "right", "ok", "okay"}
_NO_WORDS = {"no", "nope", "nah", "nahi", "na"}

_DISEASE_NEGATIVE_WORDS = {"no", "none", "nothing", "nil", "negative"}
_DISEASE_FILLER_WORDS = {"yes", "i", "do", "have", "some", "a", "few", "condition", "conditions", "issues", "problem", "problems"}

_INSURANCE_NEGATIVE_WORDS = {"no", "none"}

# Common Indian insurers, just for nicer normalization if one is mentioned.
_KNOWN_INSURERS = [
    "lic", "hdfc life", "icici prudential", "sbi life", "max life",
    "bajaj allianz", "tata aig", "star health", "care health", "niva bupa",
    "aditya birla", "kotak life", "pnb metlife",
]

_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores|k|thousand)?")
_NUMBER_RE = re.compile(r"\b(\d{1,3})\b")
_NAME_TRIGGER_RE = re.compile(
    r"(?i)(?:my name is|my name's|i am|i'm|this is|call me|change my name to|update my name to|set my name to|change name to|update name to|name is|name to)\s+([a-zA-Z ]+)"
)
_GREETING_PREFIX_RE = re.compile(r"(?i)^(hi|hello|hey)[,! ]*")
_NAME_FILLER_WORDS = {"my", "name", "is", "i", "am", "this", "hi", "hello", "hey", "call", "me"}
_NAME_NON_ANSWERS = {
    "can", "could", "would", "please", "repeat", "again", "sorry", "what", "why", "how",
    "yes", "no", "okay", "ok", "thanks", "thank", "you", "hear", "understand",
    "years", "year", "old", "yr", "yrs", "age", "aged",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "lakh", "lakhs", "crore", "crores",
    "male", "female", "man", "woman", "boy", "girl",
    "single", "married", "divorced", "widowed",
    "smoker", "tobacco", "smoke", "smoking",
    "floater", "individual", "family", "members", "member", "parents", "children",
    "budget", "premium", "coverage", "cover", "policy", "insurance", "plan",
    "looking", "want", "need", "have", "dont", "do", "not", "actually"
}
_CITY_FILLER_RE = re.compile(r"(?i)\b(i live in|i'm from|i am from|it'?s|from)\b")
_OCCUPATION_FILLER_RE = re.compile(r"(?i)\b(i work as|i am an?|i'm an?)\b")


def _tokenize(text: str) -> Set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _parse_name(text: str) -> Optional[str]:
    # Handle explicit refusals to state a name so the conversation can proceed cleanly
    text_l = text.lower().strip()
    refusal_words = {"no name", "anonymous", "skip", "don't want to say", "dont want to say", "prefer not to say", "dont tell", "don't tell"}
    if any(w in text_l for w in refusal_words):
        return "Customer"

    # Reject if text contains digits or explicit age markers like 'years old'
    if re.search(r"\b\d+\b", text) or re.search(r"\b(years|year|yr|yrs)\s+old\b", text, re.I):
        return None

    # 1. Direct trigger match
    match = _NAME_TRIGGER_RE.search(text)
    if match:
        candidate = match.group(1).strip(" .,!")
        words = [w for w in candidate.split() if w.lower() not in _NAME_FILLER_WORDS]
        if (
            words
            and all(w.isalpha() for w in words)
            and not any(w.lower() in _NAME_NON_ANSWERS for w in words)
        ):
            return " ".join(words).title()

    # 2. Fallback: strip greeting and filler words
    cleaned = _GREETING_PREFIX_RE.sub("", text).strip(" .,!")
    words = [w for w in cleaned.split() if w.lower() not in _NAME_FILLER_WORDS]
    if (
        1 <= len(words) <= 3
        and all(w.isalpha() for w in words)
        and not any(w.lower() in _NAME_NON_ANSWERS for w in words)
    ):
        return " ".join(words).title()

    return None


def normalize_stt_aliases(text: str) -> str:
    """Correct only known, context-safe STT phrase substitutions.

    The amount-unit rule deliberately requires a monetary marker, preventing a
    normal sentence such as "one black car" from becoming "one lakh car".
    """
    normalized = text
    for pattern, replacement in _STT_PHRASE_ALIASES:
        normalized = pattern.sub(replacement, normalized)
    return _AMOUNT_UNIT_ALIAS_RE.sub(
        lambda match: f"{match.group('amount')} lakh {match.group('marker')}",
        normalized,
    )


def parse_spoken_number(text: str) -> Optional[float]:
    """Parses plain numbers, Indian/Western units, and English word numbers (e.g. 'Nineteen thousand five hundred')."""
    text = normalize_stt_aliases(text)
    # Separate concatenated digits from adjacent alphabetic characters (for
    # example, '75lakh' -> '75 lakh'). Do not split word-number tokens here:
    # a partial regex match can turn 'seventy' into 'seven ty'.
    text = re.sub(r"(\d+)([a-zA-Z]+)", r"\1 \2", text)
    # Clean text to keep only words and spaces
    text = re.sub(r'[^\w\s]', ' ', text.lower()).strip()
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
        'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
    }
    multipliers = {
        'hundred': 100,
        'thousand': 1000,
        'k': 1000,
        'lakh': 100000,
        'lakhs': 100000,
        'crore': 10000000,
        'crores': 10000000
    }
    
    # Check if there are actual digits
    digit_match = re.search(r'(\d+(?:\.\d+)?)', text)
    if digit_match:
        val = float(digit_match.group(1))
        # Check multiplier
        for mult_word, mult_val in multipliers.items():
            if f" {mult_word}" in f" {text} ":
                val *= mult_val
                break
        return val
        
    words = text.split()
    total = 0
    current = 0
    parsed_any = False
    
    for word in words:
        if word in word_to_num:
            current += word_to_num[word]
            parsed_any = True
        elif word in multipliers:
            mult = multipliers[word]
            if current == 0:
                current = 1  # e.g., 'a thousand' or 'lakh'
            current *= mult
            if mult >= 1000:
                total += current
                current = 0
            parsed_any = True
            
    total += current
    return total if parsed_any else None


def _parse_age(text: str) -> Optional[int]:
    val = parse_spoken_number(text)
    if val is None:
        return None
    age = int(val)
    return age if 1 <= age <= 110 else None


def _parse_gender(text: str) -> Optional[str]:
    text_l = text.lower()
    if re.search(r"\b(female|woman|girl|f)\b", text_l):
        return "female"
    if re.search(r"\b(male|man|boy|m)\b", text_l):
        return "male"
    return None


def _parse_city(text: str) -> Optional[str]:
    cleaned = _CITY_FILLER_RE.sub("", text).strip(" .,!")
    return cleaned.title() if cleaned else None


def _parse_marital_status(text: str) -> Optional[str]:
    text_l = text.lower()
    for status in ("single", "married", "divorced", "widowed"):
        if status in text_l:
            return status
    return None


def _parse_occupation(text: str) -> Optional[str]:
    match = _OCCUPATION_FILLER_RE.search(text)
    if match:
        cleaned = _OCCUPATION_FILLER_RE.sub("", text).strip(" .,!")
        return cleaned if cleaned else None
    return None


_AMOUNT_KEYWORD_RE = re.compile(
    r"(?i)\b(rs|rupees|rupee|lakh|lakhs|crore|crores|k|thousand|thousands|hundred|hundreds|budget|cover|coverage|premium|income|amount|₹)\b"
)


def _parse_amount(text: str) -> Optional[float]:
    """Parses plain numbers, Indian/Western units, and English word amounts (e.g. 'Nineteen thousand five hundred')."""
    text = normalize_stt_aliases(text)
    val = parse_spoken_number(text)
    if val is None:
        return None
    has_keyword = bool(_AMOUNT_KEYWORD_RE.search(text))
    if val >= 500 or has_keyword:
        return val
    return None


def _parse_family_members(text: str) -> Optional[int]:
    text_l = text.lower().strip()
    if any(phrase in text_l for phrase in ("individual", "just me", "only me", "myself", "single plan", "self", "single")):
        return 1
    val = parse_spoken_number(text)
    if val is not None:
        count = int(val)
        if 1 <= count <= 20:
            return count
    family_phrases = ("family floater", "family float", "family photo", "family plan", "floater plan", "floater", "family")
    if any(phrase in text_l for phrase in family_phrases):
        return 4
    return None


def _parse_yes_no(text: str) -> Optional[bool]:
    text_l = text.lower()
    if any(phrase in text_l for phrase in ("not sure", "not certain", "don't know", "do not know", "no idea")):
        return None
    tokens = _tokenize(text)
    if tokens & _NO_WORDS:
        return False
    if tokens & _YES_WORDS:
        return True
    return None


def _parse_diseases(text: str) -> Optional[List[str]]:
    tokens = _tokenize(text)
    if tokens & _DISEASE_NEGATIVE_WORDS or "not any" in text.lower():
        return []
    parts = re.split(r",|\band\b", text, flags=re.IGNORECASE)
    diseases = [p.strip().lower() for p in parts if p.strip()]
    diseases = [d for d in diseases if d not in _DISEASE_FILLER_WORDS]
    return diseases if diseases else None


def _parse_insurer(text: str) -> Optional[str]:
    text_l = text.lower().strip()
    tokens = _tokenize(text_l)
    if tokens & _INSURANCE_NEGATIVE_WORDS or "don't have" in text_l or "dont have" in text_l or "no preference" in text_l:
        return "none"
    for insurer in _KNOWN_INSURERS:
        if insurer in text_l:
            return insurer.title()
    if any(kw in text_l for kw in ("insurer", "insurance company", "company", "preferred")):
        cleaned = text.strip(" .,!")
        return cleaned if cleaned else None
    return None


_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_EMAIL_FILLER_RE = re.compile(
    r"(?i)\b(my email address is|my email id is|my email is|my mail is|email is|email id is|send to|mail to|send it to|it is|it's|is)\b"
)
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}
_ONES = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19"
}


def _parse_email(text: str) -> Optional[str]:
    if not text:
        return None

    # 1. Direct match if standard format already
    match = _EMAIL_RE.search(text)
    if match:
        return match.group(0).lower()

    # 2. Strip conversational filler phrases
    cleaned = _EMAIL_FILLER_RE.sub("", text).strip()

    # 3. Handle spoken at-phrases like 'at the rate', 'at the rate of', 'at rate'
    cleaned = re.sub(r"(?i)\b(at the rate of|at the rate|at rate|at)\b", "@", cleaned)
    cleaned = re.sub(r"(?i)\b(dot)\b", ".", cleaned)

    # 4. Tokenize and convert spoken numbers (compound tens + ones or single digits)
    tokens = cleaned.split()
    processed_tokens = []
    i = 0
    while i < len(tokens):
        t_curr = tokens[i].lower().strip(",.!")
        if t_curr in _TENS and i + 1 < len(tokens):
            t_next = tokens[i + 1].lower().strip(",.!")
            if t_next in _ONES and isinstance(_ONES[t_next], str) and _ONES[t_next].isdigit():
                val = _TENS[t_curr] + int(_ONES[t_next])
                processed_tokens.append(str(val))
                i += 2
                continue

        if t_curr in _TENS:
            processed_tokens.append(str(_TENS[t_curr]))
        elif t_curr in _ONES:
            processed_tokens.append(str(_ONES[t_curr]))
        else:
            processed_tokens.append(tokens[i])
        i += 1

    # 5. Reconstruct tokens: collapse isolated single letters/numbers (e.g. 'B h a u' -> 'Bhau', 'g mail' -> 'gmail')
    reconstructed = ""
    for tok in processed_tokens:
        if tok in ("@", "."):
            reconstructed += tok
        elif len(tok) == 1 and (tok.isalnum() or tok in ("-", "_")):
            reconstructed += tok
        else:
            if reconstructed and not reconstructed.endswith(("@", ".")):
                reconstructed += " "
            reconstructed += tok

    normalized = reconstructed.lower().replace(" ", "")
    normalized = normalized.replace("g-mail", "gmail").replace("g.mail", "gmail")

    match = _EMAIL_RE.search(normalized)
    if match:
        return match.group(0).lower()

    return None




_BUDGET_KEYWORDS = {"budget", "premium", "spend", "pay", "cost", "afford", "annual budget", "yearly budget"}
_COVERAGE_KEYWORDS = {"coverage", "cover", "sum insured", "sum assure", "sum assured", "lakh", "lakhs", "crore", "crores"}
_INCOME_KEYWORDS = {"income", "salary", "earn", "earning", "annual income", "per year income"}


def _parse_budget(text: str) -> Optional[float]:
    text_l = text.lower()
    has_budget_kw = any(kw in text_l for kw in _BUDGET_KEYWORDS)
    has_coverage_kw = any(kw in text_l for kw in _COVERAGE_KEYWORDS)
    
    # If the text explicitly mentions coverage keywords (like "coverage", "sum insured", "lakh")
    # and does NOT mention budget keywords, this amount belongs to coverage, not budget.
    if has_coverage_kw and not has_budget_kw:
        return None
        
    return _parse_amount(text)


def _parse_coverage_required(text: str) -> Optional[float]:
    text_l = text.lower()
    has_budget_kw = any(kw in text_l for kw in _BUDGET_KEYWORDS)
    has_coverage_kw = any(kw in text_l for kw in _COVERAGE_KEYWORDS)
    
    # If the text explicitly mentions budget keywords (like "budget", "premium")
    # and does NOT mention coverage keywords, this amount belongs to budget, not coverage.
    if has_budget_kw and not has_coverage_kw:
        return None
        
    return _parse_amount(text)


def _parse_annual_income(text: str) -> Optional[float]:
    text_l = text.lower()
    has_income_kw = any(kw in text_l for kw in _INCOME_KEYWORDS)
    has_budget_kw = any(kw in text_l for kw in _BUDGET_KEYWORDS)
    has_coverage_kw = any(kw in text_l for kw in _COVERAGE_KEYWORDS)
    
    if (has_budget_kw or has_coverage_kw) and not has_income_kw:
        return None
        
    return _parse_amount(text)


_SMOKER_KEYWORDS = {"smoke", "smoker", "smoking", "tobacco", "cigarette", "cigar", "nicotine"}


def _parse_smoker(text: str) -> Optional[bool]:
    text_l = text.lower().strip()
    tokens = _tokenize(text_l)
    has_smoker_kw = bool(tokens & _SMOKER_KEYWORDS or any(kw in text_l for kw in _SMOKER_KEYWORDS))
    
    # Non-smoker topic keywords that indicate the user is answering a different question
    non_smoker_topic_kws = {"name", "call me", "this is", "years old", "year old", "budget", "coverage", "lakh", "crore", "floater", "individual"}
    has_other_topic = any(kw in text_l for kw in non_smoker_topic_kws)
    
    # If the user is speaking about another topic (like their name or age) and does NOT mention smoking/tobacco keywords,
    # reject parsing yes/no to prevent accidental smoker assignment.
    if has_other_topic and not has_smoker_kw:
        return None
        
    return _parse_yes_no(text)


# Maps a CustomerProfile field name to the parser used to extract it from
# the caller's raw (transcribed) text.
_PARSERS: Dict[str, Callable[[str], Any]] = {
    "name": _parse_name,
    "age": _parse_age,
    "gender": _parse_gender,
    "city": _parse_city,
    "marital_status": _parse_marital_status,
    "occupation": _parse_occupation,
    "annual_income": _parse_annual_income,
    "family_members": _parse_family_members,
    "parents_included": _parse_yes_no,
    "children_included": _parse_yes_no,
    "existing_diseases": _parse_diseases,
    "smoker": _parse_smoker,
    "current_insurance": _parse_insurer,
    "budget": _parse_budget,
    "coverage_required": _parse_coverage_required,
    "preferred_insurer": _parse_insurer,
    "email": _parse_email,
}

# Human-readable labels, in a sensible display order, for prompt building.
_FIELD_LABELS: Dict[str, str] = {
    "name": "Name",
    "age": "Age",
    "gender": "Gender",
    "city": "City",
    "marital_status": "Marital status",
    "occupation": "Occupation",
    "annual_income": "Annual income",
    "family_members": "Family size",
    "parents_included": "Parents included",
    "children_included": "Children included",
    "existing_diseases": "Pre-existing conditions",
    "smoker": "Smoker",
    "current_insurance": "Current insurance",
    "budget": "Budget",
    "coverage_required": "Coverage required",
    "preferred_insurer": "Preferred insurer",
    "email": "Email address",
}

_CURRENCY_FIELDS = {"annual_income", "budget", "coverage_required"}


def spell_out_email(email: str) -> str:
    """Formats an email address into spelled-out characters for clear text-to-speech pronunciation."""
    if not email or "@" not in email:
        return email or ""
    parts = email.lower().split("@", 1)
    username, domain = parts[0], parts[1]

    def spell_part(part: str) -> str:
        subparts = part.split(".")
        spelled_subparts = ["-".join(list(sub)) for sub in subparts]
        return " dot ".join(spelled_subparts)

    return f"{spell_part(username)} at {spell_part(domain)}"


@dataclass
class CustomerProfile:
    """Everything collected about the caller during this call."""

    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    city: Optional[str] = None
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    annual_income: Optional[float] = None
    family_members: Optional[int] = None
    parents_included: Optional[bool] = None
    children_included: Optional[bool] = None
    existing_diseases: Optional[List[str]] = None
    smoker: Optional[bool] = None
    current_insurance: Optional[str] = None
    budget: Optional[float] = None
    coverage_required: Optional[float] = None
    preferred_insurer: Optional[str] = None
    email: Optional[str] = None
    pending_email: Optional[str] = None
    email_confirmed: bool = False


    def is_filled(self, field_name: str) -> bool:
        """True if this field has a known answer (an empty list still counts:
        it means 'asked, and the answer was none')."""
        if field_name == "email":
            return bool(self.email and self.email_confirmed)
        value = getattr(self, field_name, None)
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        return True


    def apply_raw_answer(self, field_name: str, raw_text: str) -> bool:
        """
        Try to parse `raw_text` into a value for `field_name` and store it.
        Returns whether extraction succeeded. On failure the field is left
        untouched, so the same question will be asked again next turn.
        """
        parser = _PARSERS.get(field_name)
        if parser is None:
            return False
        value = parser(raw_text)
        if value is None:
            return False
        setattr(self, field_name, value)
        return True

    def to_summary_dict(self) -> Dict[str, str]:
        """{label: formatted value} for every filled field, for prompt building."""
        summary: Dict[str, str] = {}
        for field_name, label in _FIELD_LABELS.items():
            if self.is_filled(field_name):
                summary[label] = self._format_value(field_name, getattr(self, field_name))
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Serializes all profile fields to a dictionary for JSONB storage."""
        return asdict(self)

    @staticmethod
    def _format_value(field_name: str, value: Any) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, list):
            return ", ".join(value) if value else "none reported"
        if field_name in _CURRENCY_FIELDS:
            return format_indian_currency_for_speech(value)
        return str(value)
