import re
from collections.abc import Iterable


def map_to_groups(features_list, groups_dict):
    """
    Возвращает список названий групп, которые покрывают данные фичи.
    """
    groups_found = set()
    for f in features_list:
        for group_name, group_features in groups_dict.items():
            if f in group_features:
                groups_found.add(group_name)
                break
    return list(groups_found)


def canonicalize(s: str) -> str:
    """
    Приводит строку к каноническому виду: нижний регистр, замена разделителей на пробелы, удаление пунктуации.
    """
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[_\-\/]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


FEATURE_ALIASES = {
    "live in super": "live-in superintendent",
    "live in superintendent": "live-in superintendent",
    "live-in super": "live-in superintendent",
    "central ac": "central air conditioning",
    "central a c": "central air conditioning",
    "valet services": "valet",
    "valet service": "valet",
}


def normalize_features(all_features: Iterable[str]) -> list[str]:
    """
    Нормализует список фич:
    - пропускает None и пустые строки
    - канонизирует каждую фичу
    - применяет алиасы (с повторной канонизацией для исправления дефисов)
    - возвращает уникальные значения без сортировки
    """
    normalized = []
    for f in all_features:
        if not f:
            continue
        f = canonicalize(f)
        f = FEATURE_ALIASES.get(f, f)
        f = canonicalize(f)
        normalized.append(f)
    return list(set(normalized))


def find_substring_matches(keyword: str, all_features: Iterable[str]) -> set[str]:
    """Возвращает фичи, содержащие keyword как подстроку."""
    k = keyword.lower()
    return {f for f in all_features if k in f}


def find_word_matches(keyword: str, all_features: Iterable[str]) -> set[str]:
    """
    Возвращает фичи, содержащие keyword как отдельное слово.
    Ключевое слово канонизируется (чтобы дефисы/пунктуация не мешали, п.10),
    пустые строки пропускаются.
    """
    k = canonicalize(keyword)
    result = set()
    for f in all_features:
        if not f:
            continue
        if f" {k} " in f" {f} ":
            result.add(f)
    return result


def make_matches(all_features: Iterable[str], *keywords: str) -> set[str]:
    """Объединяет результаты substring-поиска по нескольким ключевым словам."""
    return set().union(*(find_substring_matches(k, all_features) for k in keywords))


def make_matches_filtered(
    all_features: Iterable[str], keyword: str, exclude: str | Iterable[str]
) -> set[str]:
    """
    Возвращает фичи, содержащие keyword, но не содержащие ни одного из exclude.
    exclude может быть строкой или итерируемым объектом строк.
    """
    excludes = {exclude} if isinstance(exclude, str) else set(exclude)

    return {
        f
        for f in find_substring_matches(keyword, all_features)
        if not any(ex in f for ex in excludes)
    }


def limit_words(max_words: int):
    """Возвращает функцию-фильтр: True, если в строке не больше max_words слов."""
    return lambda s: len(s.split()) <= max_words


def build_group(all_features, *keywords, filter_fn=None):
    """Строит группу фич по ключевым словам с опциональным фильтром."""
    res = make_matches(all_features, *keywords)
    if filter_fn:
        res = {f for f in res if filter_fn(f)}
    return res


def build_feature_groups(all_features: Iterable[str]) -> dict[str, set[str]]:
    """
    Строит группы признаков на основе нормализованных данных.
    Все группы строятся из нормализованных фич.
    """
    all_features = normalize_features(all_features)

    return {
        # ---- LIVING / BUILDING ----
        "laundry": make_matches(
            all_features,
            "laundry",
            "laundry room",
            "laundry facility",
            "on site laundry",
        ),
        "pre_war": make_matches(all_features, "pre war", "prewar"),
        "post_war": make_matches(all_features, "post war", "postwar"),
        "doorman": make_matches(all_features, "doorman", "concierge"),
        "security": make_matches(all_features, "security", "intercom", "video"),
        # ---- FINISHES ----
        "hardwood": make_matches(all_features, "hardwood", "wood floors"),
        "high_ceilings": make_matches(all_features, "high ceiling", "vaulted ceilings"),
        "fireplace": make_matches(all_features, "fireplace"),
        "storage": make_matches(all_features, "storage", "closet"),
        # ---- CLIMATE / LIGHT ----
        "central_ac": make_matches(
            all_features,
            "central air conditioning",
            "air conditioning",
            "central heat",
        ),
        "windows": make_matches(all_features, "natural light", "sunlight"),
        # ---- APPLIANCES ----
        "washer_dryer": make_matches_filtered(
            all_features, "washer", ["dish", "dishwasher"]
        )
        | make_matches(all_features, "dryer"),
        "dishwasher": find_substring_matches("dishwasher", all_features)
        | find_word_matches("dw", all_features),
        "appliances": make_matches(all_features, "stainless steel", "granite"),
        "kitchen": make_matches(all_features, "eat in kitchen", "open kitchen"),
        "luxury_bathroom": make_matches(all_features, "marble bath", "jacuzzi"),
        # ---- PETS ----
        "pets": make_matches(all_features, "pets", "pet friendly"),
        "no_pets": make_matches(all_features, "no pets"),
        # ---- OUTDOOR ----
        "balcony": make_matches(all_features, "balcony"),
        "terrace": make_matches(all_features, "terrace", "patio"),
        "roof_deck": make_matches(all_features, "roof deck"),
        "garden": make_matches(all_features, "garden", "yard"),
        "outdoor": make_matches(all_features, "outdoor space"),
        # ---- AMENITIES ----
        "gym": make_matches(all_features, "gym", "fitness center"),
        "pool": make_matches(all_features, "pool"),
        "parking": make_matches(all_features, "parking"),
        "elevator": make_matches(all_features, "elevator"),
        # ---- FINANCE ----
        "no_fee": make_matches(all_features, "no fee"),
        "rent_stabilized": make_matches(all_features, "rent stabilized"),
        "utilities": make_matches(all_features, "utilities included"),
        # ---- SIZE / TYPE ----
        "size": make_matches(all_features, "spacious", "large"),
        "building_type": make_matches(all_features, "brownstone", "highrise"),
    }


def get_final_features(all_features: Iterable[str]) -> list[str]:
    """
    Возвращает финальный список фич для ML-модели.
    Включает только те признаки, которые попали хотя бы в одну из групп.
    """
    groups = build_feature_groups(all_features)
    flat = set()
    for v in groups.values():
        flat.update(v)
    return sorted(flat)
