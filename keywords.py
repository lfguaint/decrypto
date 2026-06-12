from __future__ import annotations
import random
from models import KeywordPair

# 414 words — sourced from https://c-searle.neocities.org/decrypto
WORD_LIST = [
    "agent", "air", "alien", "ambulance", "angel", "angry", "apple", "aquarium",
    "arm", "arrow", "atlantis", "baby", "back", "bakery", "ball", "band",
    "bar", "bark", "bat", "battery", "bear", "bed", "bell", "belt",
    "berry", "bill", "birthday", "block", "board", "bolt", "bomb", "bond",
    "boom", "boot", "bottle", "box", "breath", "bridge", "brush", "buffalo",
    "bug", "bugle", "button", "calendar", "calf", "campaign", "cap", "car",
    "card", "carrot", "casino", "cast", "castle", "cat", "cell", "centaur",
    "centre", "chair", "change", "charge", "check", "chest", "chick", "chicken",
    "china", "chocolate", "christmas", "church", "circle", "cliff", "cloak", "cloud",
    "club", "code", "coffee", "cold", "comic", "compound", "conductor", "contract",
    "cook", "corpse", "cotton", "court", "cover", "crane", "crash", "cricket",
    "cross", "crown", "cycle", "dance", "date", "day", "death", "deck",
    "degree", "diamond", "dice", "dinosaur", "disc", "disease", "doctor", "dog",
    "draft", "dragon", "dress", "drill", "duck", "dwarf", "eagle", "embassy",
    "engine", "eye", "face", "fair", "fall", "fan", "farm", "feather",
    "fence", "field", "fighter", "figure", "file", "film", "fire", "fish",
    "flute", "fly", "foot", "force", "forest", "fork", "fountain", "fox",
    "game", "gas", "genius", "ghost", "giant", "glass", "gloom", "glove",
    "goblin", "gold", "goose", "grace", "grass", "greece", "green", "ground",
    "gun", "ham", "hand", "happy", "hawk", "head", "heart", "helicopter",
    "hole", "hood", "hook", "horde", "horn", "horse", "horseshoe", "hospital",
    "hotel", "ice", "ice cream", "internet", "iron", "ivory", "jack", "jet",
    "jupiter", "kangaroo", "ketchup", "key", "kid", "king", "kiwi", "knife",
    "knight", "lab", "lap", "laser", "laundry", "lawyer", "lead", "leprechaun",
    "life", "light", "limousine", "line", "link", "lion", "litter", "lock",
    "log", "luck", "magazine", "mail", "mammoth", "maple", "marble", "march",
    "mass", "match", "maze", "mercury", "microscope", "millionaire", "mine", "miner",
    "mint", "mischief", "missile", "model", "mole", "moon", "mount", "mountain",
    "mouse", "mouth", "muffin", "mug", "nail", "needle", "net", "night",
    "ninja", "note", "novel", "nut", "octopus", "oil", "olive", "opera",
    "orange", "organ", "palm", "pan", "pants", "paper", "parachute", "park",
    "part", "pass", "paste", "penguin", "phoenix", "piano", "pie", "pilot",
    "pin", "pipe", "pirate", "pistol", "pit", "pitch", "plane", "plastic",
    "plate", "platypus", "play", "plot", "point", "poison", "pole", "police",
    "politician", "pool", "port", "posh", "post", "pound", "press", "princess",
    "professor", "pumpkin", "pupil", "pyramid", "queen", "rabbit", "racket", "revolution",
    "ring", "risk", "robin", "robot", "rock", "romance", "root", "rose",
    "roulette", "round", "row", "ruler", "sad", "sample", "satellite", "saturn",
    "scale", "scary", "school", "scientist", "scissors", "scorpion", "screen", "scroll",
    "scuba", "seal", "server", "shadow", "shark", "shield", "ship", "shoe",
    "shop", "shot", "sink", "skyscraper", "slip", "slug", "smuggler", "snow",
    "sock", "soldier", "soul", "sound", "space", "spell", "spider", "spike",
    "spine", "spiteful", "sport", "spot", "spring", "spy", "square", "stadium",
    "staff", "star", "state", "stick", "stock", "stone", "straw", "stream",
    "string", "sub", "suit", "sun", "superhero", "swing", "switch", "sword",
    "table", "tablet", "tag", "tail", "tap", "tea", "teacher", "telescope",
    "temple", "theatre", "thief", "thumb", "tick", "tie", "time", "tissue",
    "tooth", "torch", "tower", "tractor", "train", "tree", "triangle", "trip",
    "trunk", "tube", "turkey", "undertaker", "unicorn", "vacuum", "van", "vet",
    "villain", "wake", "wall", "washer", "watch", "water", "waterfall", "wave",
    "web", "wedding", "well", "whale", "whip", "whistle", "whiteboard", "wind",
    "wish", "witch", "word", "yard", "zodiac", "zoo",
]


_MAX_REDRAWS = 1000


def generate_keyword_pairs(
    num_keywords: int = 4,
    rng: random.Random | None = None,
    seen: set | None = None,
) -> list[KeywordPair]:
    """Draw a keyword set. If `seen` is provided, guarantees the set of words
    (regardless of numbering) was never drawn before, and registers it."""
    if num_keywords > len(WORD_LIST):
        raise ValueError(
            f"num_keywords ({num_keywords}) exceeds vocabulary size ({len(WORD_LIST)})."
        )
    r = rng or random.Random()
    for _ in range(_MAX_REDRAWS):
        words = r.sample(WORD_LIST, num_keywords)
        key = frozenset(words)
        if seen is None:
            break
        if key not in seen:
            seen.add(key)
            break
    else:
        raise RuntimeError(
            f"Could not draw a new unique keyword set after {_MAX_REDRAWS} attempts "
            f"({len(seen)} sets already drawn)."
        )
    return [KeywordPair(number=i + 1, word=word) for i, word in enumerate(words)]
