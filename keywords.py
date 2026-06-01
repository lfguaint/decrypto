from __future__ import annotations
import random
from models import KeywordPair

WORD_LIST = [
    "apple", "bridge", "castle", "desert", "eclipse",
    "falcon", "garden", "harbor", "island", "jungle",
    "kitten", "lantern", "mountain", "needle", "ocean",
    "palace", "quartz", "river", "shadow", "temple",
    "umbrella", "valley", "willow", "xenon", "yellow",
    "zebra", "anchor", "balloon", "candle", "dragon",
    "emerald", "forest", "glacier", "horizon", "ivory",
    "jasmine", "kettle", "lighthouse", "marble", "nebula",
    "orbit", "pebble", "quicksand", "rainbow", "sunrise",
    "thunder", "universe", "volcano", "waterfall", "xylophone",
]


def generate_keyword_pairs(rng: random.Random | None = None) -> list[KeywordPair]:
    r = rng or random.Random()
    words = r.sample(WORD_LIST, 4)
    return [KeywordPair(number=i + 1, word=word) for i, word in enumerate(words)]
