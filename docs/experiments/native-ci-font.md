# Pinned native CI font

The native proof retains its original font checksum and absolute font path.
Ubuntu 26.04's `fonts-dejavu-extra` 2.37-8build1 has different font bytes from
the original Ubuntu 24.04 package, despite the same upstream DejaVu version.

`scripts/bootstrap_native_font.py` fetches Ubuntu's archived 2.37-8 package,
verifying its 1,947,322-byte length and SHA256 before decoding any members.
The [official package listing](https://packages.ubuntu.com/noble/all/fonts-dejavu-extra/download)
publishes package SHA256
`5502e5ccc416683e5251dc105b37be31158bc2e02133221e56524ea2a8e426fe`.
The script selects only the regular font file and its copyright/license text,
then checks each member's exact size and SHA256. It never installs the package
or executes its maintainer scripts.

CI copies the verified font to the existing historical path. Its SHA256 remains
`1d7476da447f9455a47c41adfd6ac2b1fc1b65e660f3ea4bf36128506c7e094c`.
The fetched license is retained alongside the CI installation. No historical
trainer, renderer, source binding, golden font hash or report is changed.

```sh
python3 -m unittest discover -s scripts -p 'test_bootstrap_native_font.py'
python3 scripts/bootstrap_native_font.py --destination /tmp/ocracle-native-font
```
