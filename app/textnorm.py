"""Answer/passage normaliser of the answer-hit metric.

Verbatim copy of `normalize` in the project's cleaning code (v2), vendored so this repo is self-contained.
tests/test_pipeline_local.py checks it against the original when the project code is present.
"""
import re


def normalize(text):
    """v2: every non-letter/digit is a space, so '1962 -- 1990' == '1962–1990'."""
    text = re.sub(r'<[^>]+>', ' ', text).lower()
    text = re.sub(r'[\W_]+', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())
