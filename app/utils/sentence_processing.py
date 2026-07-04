import re

from app.utils import get_nlp

_DEFAULT_ABBREVIATIONS = {
    "et al.": "ETALL",
    "i.e.": "IE",
    "e.g.": "EG",
    "etc.": "ETC",
    "vs.": "VS",
    "cf.": "CF",
    "fig.": "FIG",
    "Fig.": "FIG",
    "no.": "NO",
    "vol.": "VOL",
    "pp.": "PP",
}


def split_into_sentences(
    paragraph: str,
    min_words: int = 4,
    abbreviations: dict[str, str] | None = None,
) -> list[str]:
    abbr_map = _DEFAULT_ABBREVIATIONS if abbreviations is None else abbreviations

    protected = paragraph
    if abbr_map:
        for abbr, placeholder in abbr_map.items():
            protected = protected.replace(abbr, placeholder)

    doc = get_nlp()(protected)
    sentences: list[str] = []

    for sent in doc.sents:
        text = sent.text.strip()
        if abbr_map:
            for abbr, placeholder in abbr_map.items():
                text = text.replace(placeholder, abbr)

        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) >= min_words:
            sentences.append(text)

    return sentences
