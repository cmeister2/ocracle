# Printed language coverage

All previously printed languages remain in scope. Latin-script languages are the primary target; **non-Latin alphabets are best effort**, following the user's clarification. Phyrexian remains an optional stretch goal. Recognising some shared characters does not establish another language's coverage.

[training/languages.json](../training/languages.json) is the versioned implementation inventory, checked on 2026-10-19. [Scryfall's language catalogue](https://scryfall.com/docs/api/languages) currently lists 20 codes, including unofficial languages, and warns that older-language data is incomplete. Nineteen codes have records; Klingon currently has none. Each populated code was checked against an actual paper-card API record. This is evidence of printing, not a complete historical census or a trained-model capability claim.

| Scryfall code; printed footer code when different | Language and script/layout requirement |
| --- | --- |
| `en`, `es`→`sp`, `fr`, `de`, `it`, `pt` | English, Spanish, French, German, Italian, Portuguese: Latin text, preserving each language's diacritics and punctuation. |
| `ja`→`jp` | Japanese: Han, hiragana and katakana; horizontal and vertical fields require separate reading-order annotations. Wizards explicitly documents a [vertical Japanese Mystical Archive title bar](https://magic.wizards.com/en/news/feature/collecting-strixhaven-school-mages-2021-03-25). |
| `ko`→`kr` | Korean: Hangul and printed punctuation. |
| `ru` | Russian: Cyrillic. Historical printings remain in scope even when later sets omit the language. |
| `zhs`→`cs`; `zht`→`ct` | Simplified Chinese; Traditional Chinese: preserve the distinct printed Han forms. |
| `he` | Hebrew, including right-to-left text: [Glory](https://scryfall.com/card/pjud/11). |
| `la` | Latin: [Raging Kavu](https://scryfall.com/card/pinv/262). |
| `grc`→`ag` | Ancient Greek with polytonic marks: [Questing Phelddagrif](https://scryfall.com/card/ppls/119), plus [Cleopatra](https://scryfall.com/card/acr/272) and [Sokrates](https://scryfall.com/card/acr/273). |
| `ar` | Arabic, including joining, marks and right-to-left order: [Stone-Tongue Basilisk](https://scryfall.com/card/pody/276). |
| `sa` | Sanskrit in Devanagari, preserving vowel signs and conjuncts: [Fungal Shambler](https://scryfall.com/card/papc/100). |
| `qya` | Quenya in Tengwar letterforms: the six themed/serialized Sol Ring records. Wizards identifies the language and script in its [Middle-earth preview](https://magic.wizards.com/en/news/announcements/a-first-look-at-the-lord-of-the-rings-tales-of-middle-earth). |
| `qya`, separate `black_speech` profile | The [one-of-one One Ring](https://api.scryfall.com/cards/93de9042-cc62-4ade-8d8d-68fdbc84bfae) is also catalogued as `qya`, but Wizards identifies its language as Black Speech, using Tengwar letterforms, in the same [preview](https://magic.wizards.com/en/news/announcements/a-first-look-at-the-lord-of-the-rings-tales-of-middle-earth). Language routing must preserve this distinction. |
| `dw` | Dwarvish: Wizards confirms [five The Hobbit printings](https://magic.wizards.com/en/news/feature/collecting-the-hobbit), released August 14, 2026. Their specific script/font encoding needs confirmation. The observed [Mox Amber record](https://api.scryfall.com/cards/b89bcdd8-189e-41bb-bb3d-a53e6e6758e1) has no printed-text fields; obtain reviewed labels from pixels. |
| `ph` | **Optional stretch:** Phyrexian. Wizards documents [printed cards](https://magic.wizards.com/en/news/feature/collecting-phyrexia-all-will-be-one) and an [alphabet](https://magic.wizards.com/en/news/feature/a-breakthrough-in-phyrexian-language-and-communications). Establish glyph encoding and layout before training. |
| `tlh`→`kl` | Klingon is a catalogue entry with zero observed records. Keep it on the watch list until actual printed-card evidence establishes inclusion. |

Use one main-text language profile per card. The CLI implements explicit profile selection; it does not automatically detect language. Latin artist/collector metadata can share a recognition path across profiles. Fictional language and script are separate concepts: the Dwarven-themed 2023 Sol Rings still use Quenya, while 2026 `dw` cards form a different entry.

For acquisition, use the nullable `printed_name`, `printed_type_line`, and `printed_text` fields on the card **and each card face**, as described in [Scryfall's card-object documentation](https://scryfall.com/docs/api/cards). Confirm them visually and retain original transcriptions. Missing fields are missing labels. English Oracle text and translations cannot replace observed printed-language ground truth.

The proposed transcription standard preserves Unicode and uses NFC for evaluation by default. Record any presentation-form or private-use mapping explicitly; never reuse English ASCII folding. Keep logical text order separate from pixel reading order. Arabic/Hebrew require matching shaping and direction transforms in training and inference, including inline digits and mana symbols. Japanese vertical fields need orientation, column order, and any printed ruby annotations. Sanskrit requires conjunct-aware shaping rather than rendering isolated letters.

For fictional scripts, freeze a glyph-to-token/font mapping and version it with the checkpoint. The observed Tengwar fields use private-use characters; Phyrexian fields use an ASCII-looking font encoding. Neither is self-describing text for the current English model. Keep glyph transcription, transliteration, and translation as separate outputs. Dwarvish missing labels and uncertain mappings remain explicit work items.

Train and evaluate each profile across the font inventory, text roles, italics, clean images and strong blur. Group complete card identities and print families when splitting data. For singleton promotional languages, reserve the printed card for evaluation and train from independent script material or transferred models; report the lack of independent same-language card identities. Publish per-profile metrics and failures before marking any profile comprehensive. The inventory records seven primary Latin-script profiles, twelve best-effort non-Latin profiles, one Phyrexian stretch profile, and one unverified Klingon watch entry; it enables no additional OCR capability by itself.
