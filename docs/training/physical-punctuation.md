# Physical punctuation training crops

The default PP-OCRv6 dictionary cannot emit inverted exclamation `¡` or German
opening low quote `„`. The native training experiment can append those classes
without shifting existing indices, but procedural exposure alone does not prove
recognition on printed cards. These three real source lines provide supervised
training examples, separate from all current evaluation identities.

| Canonical identity | Language | Literal printed line |
| --- | --- | --- |
| Finishing Blow | German | `„In dieser Welt gilt: Töten oder getötet` |
| Radiant Fountain | German | `„Alle Völker schätzen einen Ort, an dem` |
| Keen Glidemaster | Spanish | `¡Salta y ya está!”.` |

All three are ordinary italic flavor text on genuine localized M21 scans.
German rows contain complete first lines. The Spanish row is the complete final
flavor line, including its closing right curly quote and outer period. Every
glyph, diacritic and crop boundary was inspected independently by two agents;
the Spanish crop excludes the preceding line's descender. No recognition output
or Oracle text supplied these labels. Card metadata was used for acquisition
and identity only; printed punctuation is retained literally in NFC.

[annotations.json](../../training/punctuation-corpus/annotations.json) freezes
transcriptions, RGB bounds, source/crop hashes and the completed independent
review. [cards.json](../../training/punctuation-corpus/cards.json) preserves the
source metadata and image URLs. Each canonical name hashes to the existing
training partition, so no validation or test identity is repurposed. The checker
also rejects overlap with the gallery, accent, rare-script, Japanese, poster and
fresh blur evaluation metadata when present.

The physical target counts are only two `„` and one `¡`. They establish positive
training exposure, not reliable glyph accuracy. No physical capital `ẞ` example
is claimed. Existing CRNN lineage includes these identities; generic Paddle
pretraining membership is unknown. They must remain training data and must not
be described as unseen evaluation examples.

Verify source-independent evidence with Python's standard library:

```sh
python3 -S training/punctuation-corpus/prepare.py --check
```

For exact fresh-source replay, use the pinned Pillow 13.0.0 training environment:

```sh
python training/punctuation-corpus/prepare.py --check --replay --download-sources
```

Downloads fail on source hash mismatch. Replay reproduces the unchanged PNG bytes
without rewriting the images or annotations. Original card text and imagery
remain Wizards of the Coast material; these fan OCR training sources are not
relicensed as MIT crate source. Data augmentation belongs in the training loop,
so these reviewed clean inputs remain fixed.
