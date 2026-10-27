"""One conservative vanilla food catalog shared by harvesting, planning and eating.

Unknown mod items, poisonous foods, suspicious stew and teleporting chorus fruit are not
automatically classified as safe rations merely because their names look edible.
"""

FOOD_PRIORITY = (
    "cooked_beef", "cooked_porkchop", "cooked_mutton", "bread", "baked_potato",
    "cooked_chicken", "cooked_rabbit", "cooked_salmon", "cooked_cod", "pumpkin_pie",
    "mushroom_stew", "rabbit_stew", "beetroot_soup", "apple", "carrot", "beetroot",
    "melon_slice", "sweet_berries", "glow_berries", "cookie", "dried_kelp", "honey_bottle",
    "golden_carrot", "beef", "porkchop", "mutton", "rabbit", "cod", "salmon",
    "tropical_fish", "potato", "chicken", "golden_apple", "enchanted_golden_apple",
)
FOOD_NAMES = frozenset(FOOD_PRIORITY)
FOOD_IDS = tuple("minecraft:" + name for name in FOOD_PRIORITY)
RAW_FOOD = frozenset(("beef", "porkchop", "mutton", "rabbit", "cod", "salmon", "chicken"))
READY_FOOD = FOOD_NAMES - RAW_FOOD


def is_edible(item: str) -> bool:
    """Bare names are accepted for planner fixtures; unknown namespaces are not inferred."""
    value = str(item)
    if ":" in value:
        namespace, value = value.split(":", 1)
        if namespace != "minecraft":
            return False
    return value in FOOD_NAMES
