`ocracle-mtg-latin-v2` is an unregistered experimental CRNN for English, French, German, Italian, Portuguese and Spanish card text.

Independent physical accents: 12/20 exact, 2.73% CER, improving V1’s 2/20 and 6.67%. French and Italian remain unreliable; this does not establish universal font or language support.

- [Training, results, limitations and reproduction](../docs/training/latin-v2.md)
- [Pinned artifact manifest](latin-v2-manifest.json)
- [Independent Rust observations](../training/latin-eval/results-v2.json)
- [English font preservation observations](../training/latin-v2-english-gallery.json)
- [Model license](LICENSE-CC-BY-SA-4.0.txt)

The model is a CC-BY-SA-4.0 derivative of Robert Knight’s `ocrs-models` recognizer. It is not selected through the runtime profile registry.
