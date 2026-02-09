# POS-Filtered LDA Evaluation

## Topic 3 Top Words (Before vs After POS Filtering)
- Positive before: coffee, breakfast, brunch, egg, cream, toast, sweet, french, tea, chocolate
- Positive after: bar, drink, beer, night, table, happy, hour, great, selection, friend
- Negative before: come, table, wait, service, ask, drink, time, order, minute, go
- Negative after: cheese, sandwich, good, fry, chicken, burger, salad, order, place, time

## Coherence (c_v)
- Positive: c_v 0.4300 -> 0.4851 (delta +0.0551).
- Negative: c_v 0.4246 -> 0.4314 (delta +0.0069).

## What Improved and Trade-offs
- Topic 3 reduced verb-like words in positive split (0 -> 0) and negative split (3 -> 0).
- Topic 3 lexical overlap before/after is limited (positive 0/10, negative 2/10), which indicates stronger vocabulary reshaping toward entity/descriptor words.
- Narrative verbs were reduced mainly in negative Topic 3 (for example, come/wait/ask/go), while positive Topic 3 was already noun/adjective-heavy.
- Trade-off: POS filtering can drop operational action cues and shift emphasis to object/quality descriptors (positive new examples: bar, drink, beer; negative new examples: cheese, sandwich, good).
