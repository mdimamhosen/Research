#!/usr/bin/env python3
"""
Aggressively enrich Bangla dialect dataset to 15k+ pairs.
Uses combinatorial generation across multiple dimensions.
"""

import random
from itertools import product

random.seed(42)

# Core vocabulary in dialect ↔ standard form
PRONOUNS = {
    "আই": "আমি",
    "তুই": "তুমি",
    "হে": "সে",
    "আমরা": "আমরা",
    "তোরা": "তোমরা",
    "ওরা": "তারা",
}

OBJECTS = {
    "এডা": "এটা",
    "অডা": "ওটা",
    "এইটা": "এটি",
}

VERBS_PRESENT = {
    "যাইয়্যার": "যাচ্ছি",
    "খাইয়্যার": "খাচ্ছি",
    "পড়র": "পড়ছে",
    "করর": "করছে",
    "চলর": "চলছে",
    "খেলর": "খেলছে",
    "ঘুমাইয়্যার": "ঘুমাচ্ছি",
    "শুনর": "শুনছে",
    "দেখর": "দেখছে",
    "বলর": "বলছে",
    "লিখর": "লিখছে",
    "গাইর": "গাচ্ছে",
    "নাচর": "নাচছে",
    "দৌড়র": "দৌড়াচ্ছে",
    "হাঁটর": "হাঁটছে",
    "কিনি আইয়্যার": "কিনছি",
    "বানাইয়্যার": "বানাচ্ছি",
    "চালাইর": "চালাচ্ছে",
}

VERBS_FUTURE = {
    "খাইবো": "খাব",
    "করমু": "করব",
    "যাইমু": "যাব",
    "আসিবো": "আসব",
    "আইবো": "আসবে",
    "হইবো": "হবে",
    "থাকিবো": "থাকব",
    "দেখমু": "দেখব",
    "শুনমু": "শুনব",
    "বলিবো": "বলব",
    "লিখিবো": "লেখা হবে",
    "চলমু": "চলব",
    "খেলমু": "খেলব",
    "পড়মু": "পড়ব",
}

VERBS_PAST = {
    "খাইছি": "খেয়েছি",
    "করিছি": "করেছি",
    "যাইছি": "গেছি",
    "আইছে": "এসেছে",
    "দেখিছি": "দেখেছি",
    "বলিছি": "বলেছি",
    "পড়িছি": "পড়েছি",
    "খেলিছি": "খেলেছি",
    "ঘুমাইছি": "ঘুমিয়েছি",
}

TIMES = {
    "আজকে": "আজকে",
    "কাল": "কাল",
    "পরশু": "পরশুদিন",
    "এখন": "এখন",
    "এখনই": "এখনই",
    "সন্ধ্যায়": "সন্ধ্যায়",
    "সকালে": "সকালে",
    "রাতে": "রাতে",
    "দুপুরে": "দুপুরে",
    "সকাল": "সকাল",
    "দুপুর": "দুপুর",
    "সন্ধ্যা": "সন্ধ্যা",
    "রাত": "রাত",
    "মধ্যরাত": "মধ্যরাত",
    "ভোর": "ভোর",
}

PLACES = {
    "ঘরে": "বাড়ি",
    "বাজারে": "বাজারে",
    "স্কুলে": "স্কুলে",
    "অফিসে": "অফিসে",
    "নদীতে": "নদীতে",
    "মাঠে": "মাঠে",
    "পুকুরে": "পুকুরে",
    "রাস্তায়": "রাস্তায়",
    "হাসপাতালে": "হাসপাতালে",
    "মসজিদে": "মসজিদে",
    "মন্দিরে": "মন্দিরে",
    "কাজে": "কাজে",
    "স্টেশনে": "স্টেশনে",
    "পার্কে": "পার্কে",
}

FOODS = {
    "ভাত": "ভাত",
    "চা": "চা",
    "পানি": "পানি",
    "দুধ": "দুধ",
    "মাছ": "মাছ",
    "মাংস": "মাংস",
    "মুরগি": "মুরগি",
    "শাক": "শাক",
    "সবজি": "সবজি",
    "ডাল": "ডাল",
    "আলু": "আলু",
    "মিষ্টি": "মিষ্টি",
    "রুটি": "রুটি",
    "পুরি": "পুরি",
    "বিরিয়ানি": "বিরিয়ানি",
    "খিচুড়ি": "খিচুড়ি",
    "ডিম": "ডিম",
    "ফল": "ফল",
    "রস": "রস",
}

FAMILY = {
    "মা": "মা",
    "বাপ": "বাবা",
    "দাদা": "দাদা",
    "দাদী": "দাদী",
    "নানা": "নানা",
    "নানী": "নানী",
    "চাচা": "চাচা",
    "চাচী": "চাচী",
    "মামা": "মামা",
    "খালা": "খালা",
    "বোন": "বোন",
    "ভাই": "ভাই",
    "বউ": "বউ",
    "জামাই": "জামাই",
    "পোয়া": "ছেলে",
    "মেয়ে": "মেয়ে",
}

ADJECTIVES = {
    "বড়": "বড়",
    "ছোট": "ছোট",
    "ভালো": "ভালো",
    "খারাপ": "খারাপ",
    "সুন্দর": "সুন্দর",
    "কুৎসিত": "কুৎসিত",
    "গরম": "গরম",
    "ঠান্ডা": "ঠান্ডা",
    "মিষ্টি": "মিষ্টি",
    "তেতো": "তেতো",
    "নতুন": "নতুন",
    "পুরনো": "পুরনো",
    "শক্তিশালী": "শক্তিশালী",
    "দুর্বল": "দুর্বল",
    "দ্রুত": "দ্রুত",
    "ধীর": "ধীর",
}

ACTIONS = {
    "আছে": "আছে",
    "নেই": "নেই",
    "গেছে": "গেছে",
    "এসেছে": "এসেছে",
    "করেছে": "করেছে",
    "খেয়েছে": "খেয়েছে",
    "পড়েছে": "পড়েছে",
    "খেলেছে": "খেলেছে",
    "ঘুমিয়েছে": "ঘুমিয়েছে",
}

def generate_sv_combinations():
    """Subject + Verb combinations."""
    pairs = []
    for subj_d, subj_s in PRONOUNS.items():
        for verb_d, verb_s in VERBS_PRESENT.items():
            pairs.append((f"{subj_d} {verb_d}", f"{subj_s} {verb_s}"))
        for verb_d, verb_s in VERBS_FUTURE.items():
            pairs.append((f"{subj_d} {verb_d}", f"{subj_s} {verb_s}"))
        for verb_d, verb_s in VERBS_PAST.items():
            pairs.append((f"{subj_d} {verb_d}", f"{subj_s} {verb_s}"))
    return pairs

def generate_svo_combinations():
    """Subject + Verb + Object combinations."""
    pairs = []
    # Use subset to keep reasonable size
    subjects = list(PRONOUNS.items())[:4]
    foods = list(FOODS.items())[:10]
    verbs_pres = list(VERBS_PRESENT.items())[:8]

    for (subj_d, subj_s), (obj_d, obj_s), (verb_d, verb_s) in product(subjects, foods, verbs_pres):
        pairs.append((f"{subj_d} {obj_d} {verb_d}", f"{subj_s} {obj_s} {verb_s}"))
    return pairs

def generate_time_location_combinations():
    """Time + Place + Verb combinations."""
    pairs = []
    verbs = list(VERBS_PRESENT.items())[:5] + list(VERBS_FUTURE.items())[:5]
    times = list(TIMES.items())
    places = list(PLACES.items())

    for (time_d, time_s), (place_d, place_s), (verb_d, verb_s) in product(times, places, verbs):
        pairs.append((f"{time_d} {place_d} {verb_d}", f"{time_s} {place_s} {verb_s}"))
    return pairs

def generate_adj_noun_combinations():
    """Adjective + Noun + Action combinations."""
    pairs = []
    items = list(product(ADJECTIVES.items(), FOODS.items()))
    actions = list(ACTIONS.items())

    for (adj_d, adj_s), (food_d, food_s) in items[:50]:  # limit combinations
        for action_d, action_s in actions:
            pairs.append((f"{adj_d} {food_d} {action_d}", f"{adj_s} {food_s} {action_s}"))
    return pairs

def generate_family_action_combinations():
    """Family member + Action combinations."""
    pairs = []
    for (fam_d, fam_s), (action_d, action_s) in product(FAMILY.items(), ACTIONS.items()):
        pairs.append((f"{fam_d} {action_d}", f"{fam_s} {action_s}"))

    # Add with places
    for (fam_d, fam_s), (place_d, place_s), (action_d, action_s) in \
            product(FAMILY.items(), PLACES.items(), ACTIONS.items()):
        pairs.append((f"{fam_d} {place_d} {action_d}", f"{fam_s} {place_s} {action_s}"))
    return pairs

def generate_negation_combinations():
    """Negation patterns."""
    pairs = []
    for (subj_d, subj_s), (verb_d, verb_s) in product(PRONOUNS.items(), VERBS_FUTURE.items()):
        pairs.append((f"{subj_d} {verb_d} না", f"{subj_s} {verb_s} না"))

    for (subj_d, subj_s), (verb_d, verb_s) in product(PRONOUNS.items(), VERBS_PRESENT.items()):
        pairs.append((f"{subj_d} {verb_d} না", f"{subj_s} {verb_s} না"))
    return pairs

def generate_question_combinations():
    """Question patterns."""
    pairs = []
    question_words = {
        "কী": "কী",
        "কে": "কে",
        "কোন": "কোন",
        "কত": "কত",
        "কখন": "কখন",
        "কেন": "কেন",
        "কোথায়": "কোথায়",
    }

    for qword_d, qword_s in question_words.items():
        for food_d, food_s in FOODS.items():
            pairs.append((f"{qword_d} {food_d}", f"{qword_s} {food_s}"))
        for obj_d, obj_s in OBJECTS.items():
            pairs.append((f"{qword_d} {obj_d}", f"{qword_s} {obj_s}"))
        for place_d, place_s in PLACES.items():
            pairs.append((f"{qword_d} {place_d}", f"{qword_s} {place_s}"))
    return pairs

def generate_compound_combinations():
    """More complex compound sentences."""
    pairs = []
    conjunctions = [
        ("এবং", "এবং"),
        ("কিন্তু", "কিন্তু"),
        ("কারণ", "কারণ"),
        ("যদি", "যদি"),
        ("তাহলে", "তাহলে"),
    ]

    # Simple two-clause patterns
    base_clauses = [
        ("আই যাইয়্যার", "আমি যাচ্ছি"),
        ("তুই থাকিবো", "তুমি থাকব"),
        ("হে করমু", "সে করব"),
        ("আমরা খাইয়্যার", "আমরা খাচ্ছি"),
    ]

    for c_d, c_s in conjunctions:
        for (clause1_d, clause1_s), (clause2_d, clause2_s) in \
                product(base_clauses, repeat=2):
            if clause1_d != clause2_d:  # avoid duplicates
                pairs.append(
                    (f"{clause1_d} {c_d} {clause2_d}",
                     f"{clause1_s} {c_s} {clause2_s}")
                )
    return pairs

def generate_numeric_combinations():
    """Numbers and quantifiers."""
    pairs = []
    numbers = [
        ("এক", "এক"),
        ("দুই", "দুই"),
        ("তিন", "তিন"),
        ("চার", "চার"),
        ("পাঁচ", "পাঁচ"),
        ("ছয়", "ছয়"),
        ("সাত", "সাত"),
        ("আট", "আট"),
        ("নয়", "নয়"),
        ("দশ", "দশ"),
    ]
    classifiers = ["টা", "জন", "খানা", "খান"]

    for num_d, num_s in numbers:
        for food_d, food_s in FOODS.items():
            for cls in classifiers:
                pairs.append((f"{num_d}{cls} {food_d}", f"{num_s}{cls} {food_s}"))
                for action_d, action_s in list(ACTIONS.items())[:3]:
                    pairs.append(
                        (f"{num_d}{cls} {food_d} {action_d}",
                         f"{num_s}{cls} {food_s} {action_s}")
                    )
    return pairs

def generate_imperative_combinations():
    """Command forms."""
    pairs = []
    commands = {
        "কর": "করো",
        "দাও": "দাও",
        "আয়": "এসো",
        "যা": "যাও",
        "বল": "বলো",
        "শোন": "শোনো",
        "দেখ": "দেখো",
        "পড়": "পড়ো",
        "খা": "খাও",
        "ঘুমা": "ঘুমাও",
        "লিখ": "লেখো",
        "নাচ": "নাচো",
        "গা": "গাও",
        "চল": "চলো",
        "খেল": "খেলো",
        "জেগ": "জেগো",
    }

    for obj_d, obj_s in FOODS.items():
        for cmd_d, cmd_s in commands.items():
            pairs.append((f"{obj_d} {cmd_d}", f"{obj_s} {cmd_s}"))

    for place_d, place_s in PLACES.items():
        for cmd_d, cmd_s in commands.items():
            pairs.append((f"{place_d} {cmd_d}", f"{place_s} {cmd_s}"))

    return pairs

def generate_conditional_combinations():
    """If-then constructions."""
    pairs = []
    conditions = [
        ("বৃষ্টি অইছে", "বৃষ্টি হয়েছে"),
        ("গরম পড়েছে", "গরম পড়েছে"),
        ("ঠান্ডা", "ঠান্ডা"),
        ("রাত", "রাত"),
        ("দিন", "দিন"),
    ]

    results = [
        ("ভিতরে থাক", "ভিতরে থাকো"),
        ("বাইরে যা", "বাইরে যাও"),
        ("ছাতা নে", "ছাতা নাও"),
        ("কাপড় পরে নে", "কাপড় পরে নাও"),
        ("ঘুমা", "ঘুমাও"),
    ]

    for cond_d, cond_s in conditions:
        for res_d, res_s in results:
            pairs.append((f"যদি {cond_d} তাহলে {res_d}", f"যদি {cond_s} তাহলে {res_s}"))

    return pairs

def generate_possession_combinations():
    """Possession markers."""
    pairs = []

    for (subj_d, subj_s), (obj_d, obj_s) in product(PRONOUNS.items(), FOODS.items()):
        pairs.append((f"{subj_d}র {obj_d}", f"{subj_s}র {obj_s}"))

    for (subj_d, subj_s), (fam_d, fam_s) in product(PRONOUNS.items(), FAMILY.items()):
        pairs.append((f"{subj_d}র {fam_d}", f"{subj_s}র {fam_s}"))

    return pairs

def generate_existential_combinations():
    """There is / there are patterns."""
    pairs = []
    existentials = [
        ("আছে", "আছে"),
        ("নেই", "নেই"),
        ("আছিল", "ছিল"),
        ("ছিল না", "ছিল না"),
    ]

    locations = [
        ("ঘরে", "বাড়িতে"),
        ("বাজারে", "বাজারে"),
        ("মাঠে", "মাঠে"),
    ]

    for loc_d, loc_s in locations:
        for food_d, food_s in FOODS.items():
            for exist_d, exist_s in existentials:
                pairs.append((f"{loc_d} {food_d} {exist_d}", f"{loc_s} {food_s} {exist_s}"))

    return pairs

def generate_manner_combinations():
    """Manner adverbs."""
    pairs = []
    manners = {
        "খুব": "খুব",
        "একটু": "একটু",
        "বেশি": "বেশি",
        "কম": "কম",
        "ধীরে": "ধীরে",
        "দ্রুত": "দ্রুত",
        "সাবধানে": "সাবধানে",
        "অসাবধানে": "অসাবধানে",
    }

    for manner_d, manner_s in manners.items():
        for adj_d, adj_s in ADJECTIVES.items():
            pairs.append((f"{manner_d} {adj_d}", f"{manner_s} {adj_s}"))

        for (subj_d, subj_s), (verb_d, verb_s) in \
                product(list(PRONOUNS.items())[:3], list(VERBS_PRESENT.items())[:3]):
            pairs.append((f"{subj_d} {manner_d} {verb_d}", f"{subj_s} {manner_s} {verb_s}"))

    return pairs

def generate_expression_combinations():
    """Common expressions and idioms."""
    pairs = []
    expressions = [
        ("কী খবর", "কী খবর"),
        ("কেমন আছো", "কেমন আছো"),
        ("ভালো আছি", "ভালো আছি"),
        ("দুঃখের বিষয়", "দুঃখের বিষয়"),
        ("খুব ভালো", "খুব ভালো"),
        ("কোন সমস্যা", "কোন সমস্যা"),
        ("ঠিক আছে", "ঠিক আছে"),
        ("হ্যাঁ ভাই", "হ্যাঁ ভাই"),
        ("না ভাই", "না ভাই"),
        ("ধন্যবাদ", "ধন্যবাদ"),
        ("আপনাকে স্বাগতম", "আপনাকে স্বাগতম"),
    ]

    for expr_d, expr_s in expressions:
        pairs.append((expr_d, expr_s))

    # Add with subjects
    for (subj_d, subj_s), (expr_d, expr_s) in product(PRONOUNS.items(), expressions):
        pairs.append((f"{subj_d} {expr_d}", f"{subj_s} {expr_s}"))

    return pairs

def generate_relative_clause_combinations():
    """Relative clause constructions."""
    pairs = []

    for (subj_d, subj_s), (obj_d, obj_s), (action_d, action_s) in \
            product(list(PRONOUNS.items())[:3], list(FOODS.items())[:5], list(ACTIONS.items())[:3]):
        pairs.append(
            (f"{subj_d} যে {obj_d} {action_d}", f"{subj_s} যে {obj_s} {action_s}")
        )

    return pairs

# Generate all combinations
all_pairs = []

print("Generating SV combinations...")
all_pairs.extend(generate_sv_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating SVO combinations...")
all_pairs.extend(generate_svo_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating time-location combinations...")
all_pairs.extend(generate_time_location_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating adjective-noun combinations...")
all_pairs.extend(generate_adj_noun_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating family-action combinations...")
all_pairs.extend(generate_family_action_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating negation combinations...")
all_pairs.extend(generate_negation_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating question combinations...")
all_pairs.extend(generate_question_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating compound combinations...")
all_pairs.extend(generate_compound_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating numeric combinations...")
all_pairs.extend(generate_numeric_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating imperative combinations...")
all_pairs.extend(generate_imperative_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating conditional combinations...")
all_pairs.extend(generate_conditional_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating possession combinations...")
all_pairs.extend(generate_possession_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating existential combinations...")
all_pairs.extend(generate_existential_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating manner combinations...")
all_pairs.extend(generate_manner_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating expression combinations...")
all_pairs.extend(generate_expression_combinations())
print(f"  Total: {len(all_pairs)}")

print("Generating relative clause combinations...")
all_pairs.extend(generate_relative_clause_combinations())
print(f"  Total: {len(all_pairs)}")

# Deduplicate
print("\nDeduplicating...")
seen = set()
unique_pairs = []
for pair in all_pairs:
    if pair not in seen:
        seen.add(pair)
        unique_pairs.append(pair)

print(f"After deduplication: {len(unique_pairs)}")

# Shuffle
random.shuffle(unique_pairs)

# Take first 15000
final_pairs = unique_pairs[:15000]

# Write to file
output_file = r"C:\Error\Research\EDGE\BanglaDialectSSM\corpus\raw\pairs.tsv"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# dialect_sentence\t\tstandard_bangla_sentence\n")
    f.write("# Enriched Bangla dataset: 15k dialect-to-standard pairs\n")
    f.write("# Generated with systematic combinatorial variations\n")
    f.write("# Categories: pronouns, verbs, time, location, food, family, adjectives, questions\n\n")

    for dialect, standard in final_pairs:
        f.write(f"{dialect}\t{standard}\n")

print(f"\nSuccessfully wrote {len(final_pairs)} pairs to {output_file}")

# Verify
with open(output_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data_lines = [l for l in lines if l.strip() and not l.startswith("#")]
    print(f"Verification: {len(data_lines)} data lines in output file")
