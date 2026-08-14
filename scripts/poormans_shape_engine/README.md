# poormans shape engine

A deterministic search engine for compact mathematical terminal marks. It samples eight shape families, renders each candidate at high resolution, downsamples to the real terminal grid, scores the final mark, normalizes family score distributions, and greedily removes visual near-duplicates.

Families: Clifford, De Jong, Hopalong, Gumowski-Mira, Ikeda, two-harmonic Lissajous/Fourier curves, hypotrochoids, and the superformula.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 generate_marks.py --out /tmp/marks --seed poormans-hpc
xdg-open /tmp/marks/index.html
```

A faster taste test:

```bash
python3 generate_marks.py --out /tmp/marks-fast --per-family 20 --iteration-scale 0.35
```

A denser final search:

```bash
python3 generate_marks.py --out /tmp/marks-final --per-family 160 --iteration-scale 1.6 --count 32
```

Useful knobs:

```text
--families clifford,dejong,hopalong,gumowski_mira,ikeda,lissajous,hypotrochoid,superformula
--width 60 --height 30
--supersample 4
--max-similarity 0.89
--family-balance 0.35
--max-per-family 0
--no-family-seeding
--no-pca
```

The output directory contains `index.html`, `manifest.json`, and shaded/binary `.txt` plus `.json` files for every selected mark.
