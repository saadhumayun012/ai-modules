try:
    import spacy  # type: ignore
except ImportError:  # pragma: no cover
    spacy = None

_NLP = None


def get_nlp():
    global _NLP

    if _NLP is not None:
        return _NLP

    if spacy is None:
        raise RuntimeError(
            "spaCy is not installed. Install it and download en_core_web_sm."
        )

    try:
        _NLP = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "spaCy model en_core_web_sm is not available. Run: python -m spacy download en_core_web_sm"
        ) from exc

    return _NLP
