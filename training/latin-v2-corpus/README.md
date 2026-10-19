This is the frozen data snapshot for the unregistered Latin V2 CRNN experiment.

The corpus has 43,145 lines: parent English/V1 data, 101 reviewed localized titles, 20 eligible reviewed body lines, new synthetic card text and explicit procedural glyph practice. All 63 discovered English fallback images were rejected. Canonical card identities retain one split across languages and printings.

- [Training, results, provenance and reproduction](../../docs/training/latin-v2.md)
- [Frozen counts and manifest hash](summary.json)
- [Physical versus synthetic character coverage](coverage.json)
- [All reviewed title decisions](titles-reviewed.json)
- [Body annotations](body-lines.json)
- [Reviewed contact sheets](contact-sheets/manifest.json)
- [Metadata acquisition selection](selection.json)
- [CLDR exemplar sources](orthography.json)
- [Pinned font hashes](fonts.json)
- [DejaVu font terms](DEJAVU-LICENSE.txt)
- [Verified deterministic data replay](../latin-v2-reproduction-validation.json)

Opt was removed from V2 updates but exists in the V1 initializer. The independent Latin accent test and five rare-script singletons remain disjoint throughout the known training lineage. Physical examples are sparse; glyph-practice coverage is not a physical accuracy claim.
