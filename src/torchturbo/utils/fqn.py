"""Fully qualified module and parameter name matching."""

import re


def match_fqn(pattern: str, fqn: str) -> bool:
    """Return whether an FQN matches a pattern.

    The pattern must match the entire FQN. ``*`` matches one or more characters
    other than ``.``, so it stays inside a single segment. ``**`` can also
    match ``.``, so it can span multiple segments.

    Examples:
        ``*`` stays inside one segment, and may match part of it:

        >>> match_fqn("layers.*.weight", "layers.0.weight")
        True

        ``**`` may match across multiple segments:

        >>> match_fqn("layers.**.weight", "layers.0.attention.q_proj.weight")
        True
    """
    regex = re.escape(pattern)
    regex = regex.replace(r"\*\*", r".*")
    regex = regex.replace(r"\*", r"[^.]+")
    return bool(re.match(f"^{regex}$", fqn))
