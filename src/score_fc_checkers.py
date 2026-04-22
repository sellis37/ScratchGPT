import re
from typing import Any, Dict, Tuple

def _get_arg_str(parsed_json: Any, param_name: str) -> str:
    if isinstance(parsed_json, dict) and param_name in parsed_json:
        v = parsed_json[param_name]
        return "" if v is None else str(v)
    return ""

def spaces_in_between_checker(value: str, N: int) -> int:
    # Must have exactly N spaces between every pair of words.
    # Interpreting strictly: if there are multiple words, separators must be exactly N spaces.
    # Single word is valid.
    value = value.strip()
    if not value:
        return 0
    parts = re.split(r"\s+", value)
    if len(parts) <= 1:
        return 1
    # Reconstruct with exact N spaces and compare after normalizing ends
    expected = (" " * N).join(parts)
    return 1 if expected == value else 0

def letter_frequency_checker(value: str, letter: str, N: int, comparison_option: str = "exactly") -> int:
    cnt = sum(1 for ch in value.lower() if ch == letter.lower())
    if comparison_option == "exactly":
        return 1 if cnt == N else 0
    if comparison_option == "at most":
        return 1 if cnt <= N else 0
    if comparison_option == "at least":
        return 1 if cnt >= N else 0
    return -1

def n_commas_checker(value: str, N: int, comparison_option: str = "at most") -> int:
    cnt = value.count(",")
    if comparison_option == "exactly":
        return 1 if cnt == N else 0
    if comparison_option == "at most":
        return 1 if cnt <= N else 0
    if comparison_option == "at least":
        return 1 if cnt >= N else 0
    return -1

def score_format_checker(parsed_json: Any, meta: Dict[str, Any]) -> Tuple[int, str]:
    """
    Returns: (ok_flag, checker_name)
    ok_flag: 1 pass, 0 fail, -1 unknown/unscored
    """
    checker = meta.get("format_checker")
    param = meta.get("chosen_param")

    if not isinstance(checker, dict) or not param:
        return -1, "none"

    name = checker.get("name", "")
    args = checker.get("args", {}) or {}
    v = _get_arg_str(parsed_json, param)

    if name == "SpacesInBetweenChecker":
        N = int(args.get("N", 6))
        return spaces_in_between_checker(v, N), name

    if name == "LetterFrequencyChecker":
        letter = str(args.get("letter", "n"))
        N = int(args.get("N", 0))
        comp = str(args.get("comparison_option", "exactly"))
        return letter_frequency_checker(v, letter, N, comp), name

    if name == "NCommasChecker":
        N = int(args.get("N", 0))
        comp = str(args.get("comparison_option", "at most"))
        return n_commas_checker(v, N, comp), name

    return -1, name