"""Speech-friendly formatting for Indian currency amounts used in LLM prompts."""

from __future__ import annotations


_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _under_thousand(number: int) -> str:
    parts = []
    if number >= 100:
        parts.append(f"{_ONES[number // 100]} hundred")
        number %= 100
        if number:
            parts.append("and")
    if number >= 20:
        tens, ones = divmod(number, 10)
        parts.append(_TENS[tens] if not ones else f"{_TENS[tens]}-{_ONES[ones]}")
    elif number:
        parts.append(_ONES[number])
    return " ".join(parts)


def format_indian_currency_for_speech(amount: float | int) -> str:
    """Return an exact whole-rupee amount in natural Indian English.

    Examples: ``7_000_000 -> 'seventy lakh rupees'`` and
    ``46_302 -> 'forty-six thousand three hundred and two rupees'``.
    """
    rupees = int(round(float(amount)))
    if rupees == 0:
        return "zero rupees"
    if rupees < 0:
        return f"minus {format_indian_currency_for_speech(-rupees)}"

    parts = []
    for unit_value, unit_name in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        quantity, rupees = divmod(rupees, unit_value)
        if quantity:
            parts.append(f"{_under_thousand(quantity)} {unit_name}")
    if rupees:
        parts.append(_under_thousand(rupees))
    return f"{' '.join(parts)} rupees"
