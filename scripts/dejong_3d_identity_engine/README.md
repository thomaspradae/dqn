# De Jong 3D identity engine

This package builds a deterministic 3D identity system around one canonical De Jong logo.

Canonical source:

- family: `dejong`
- seed: `poormans-hpc:dejong:00012`
- rank: `9`
- raw score: `8.470892`
- selection score: `9.039588`
- params:
  - `a = 1.1504140582765`
  - `b = -2.3598824892870325`
  - `c = 2.223958296969601`
  - `d = -1.9009456070297253`

## What it does

1. Regenerates the exact 2D De Jong point cloud from those fixed parameters.
2. Turns that point cloud into a normalized master density field.
3. Lifts the ordered 2D orbit into 3D in several different derivation modes.
4. Applies deterministic 3D rotation, twist, and projection derived from a code/hash.
5. Searches a deterministic neighborhood per code and keeps the best identity-preserving variant.
6. Renders the result to terminal-density art and exports an HTML gallery plus metadata.

## 3D derivation modes

- `orbit_embedding`: assigns one delayed-orbit Z coordinate to each De Jong point. It creates genuine 3D structure without cloning the canonical image, while preserving the exact canonical XY projection from the front.
- `surface`: density-driven relief surface with layered volume.
- `contour_stack`: topographic slice stack derived from density contours.
- `ribbon_skeleton`: trajectory-based ribbon/tube interpretation of the attractor path.
- `shell_relief`: inflated embossed shell using the canonical silhouette as a mother-form.

`orbit_embedding` is the current architectural candidate for the main identity
animation. The older modes remain available as deliberate relief, contour, and
ribbon effects, but they duplicate points into layers and should not be mistaken
for a single 3D orbit.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 generate_identity_variants.py --out /tmp/dejong3d_gallery --seed poormans-hpc --count 16
xdg-open /tmp/dejong3d_gallery/index.html
```

Restrict to one or more modes:

```bash
python3 generate_identity_variants.py --out /tmp/dejong3d_shell --modes shell_relief
python3 generate_identity_variants.py --out /tmp/dejong3d_mix --modes surface ribbon_skeleton shell_relief
python3 generate_identity_variants.py --out /tmp/dejong3d_orbit --modes orbit_embedding
```

## Notes

- `canonical` is always included as the first output.
- every other code is deterministic; the same code always yields the same chosen variant.
- the engine is 3D internally, but the exported mark is a 2D projection of that 3D form.
- if you want more exploration per code, increase `--candidates-per-code`.

## Animated identity transition

The animation layer treats the canonical logo as frame zero and a deterministic 3D descendant as the final frame.
In this repository, the default canonical terminal frame is also locked against
`../poormans_shape_engine/selected_main/poormans_hpc_main_shade.txt`, which is
the selected De Jong candidate #009. That file is the visual source of truth for
the current poormans HPC mark.

For a node or arbitrary code:

```bash
python3 animate_identity.py node:ofi1
```

The default motion is:

1. hold the exact canonical 2D De Jong mark briefly;
2. introduce depth;
3. begin camera rotation;
4. bring in twist/ripple detail slightly later;
5. end on the exact deterministic 3D target selected for that code.

Useful controls:

```bash
python3 animate_identity.py node:ofi1 --seconds 2 --fps 24
python3 animate_identity.py node:ofi1 --pingpong --loop 3
python3 animate_identity.py node:ofi1 --style binary
python3 animate_identity.py node:ofi1 --mode contour_stack
python3 animate_identity.py node:ofi1 --mode orbit_embedding
python3 animate_identity.py node:ofi1 --profile hero
python3 animate_identity.py node:ofi1 --axis-gizmo
```

Profiles:

- `clean`: startup-safe surface motion, 48 frames at 24 FPS.
- `hero`: stronger lighting and more dramatic contour-stack defaults, 72 frames at 24 FPS.
- `loop`: milder motion intended for ping-pong / idle use, 56 frames at 20 FPS.

Showcase examples:

```bash
python3 animate_identity.py --example surface_clean
python3 animate_identity.py --example contour_stack_hero
python3 animate_identity.py --example ribbon_skeleton_expressive
```

Export all text frames plus a self-playing browser preview:

```bash
python3 animate_identity.py node:ofi1 \
  --frames 49 --fps 24 \
  --export /tmp/ofi1-animation \
  --no-play

xdg-open /tmp/ofi1-animation/index.html
```

Export the three local examples:

```bash
python3 animate_identity.py --example surface_clean --export examples/surface_clean --no-play
python3 animate_identity.py --example contour_stack_hero --export examples/contour_stack_hero --no-play
python3 animate_identity.py --example ribbon_skeleton_expressive --export examples/ribbon_skeleton_expressive --no-play
```

Camera diagnostic overlay:

```bash
python3 animate_identity.py node:ofi1 \
  --profile clean \
  --frames 72 \
  --fps 20 \
  --intro-hold 1.0 \
  --hold 1.4 \
  --candidates 1 \
  --axis-gizmo
```

The overlay draws a tiny rotation gizmo at the center of the mark:

- red `X/x` is the local X axis;
- green `Y/y` is the local Y axis;
- blue `Z/z` is the local Z axis;
- white `+` is the origin.

At the front-on canonical frame, the Z axis points out of the terminal plane and
collapses near the origin. As yaw/pitch change, the blue axis should separate
from the center. If it barely separates, the animation path is too conservative
even if the attractor texture is changing.

The browser exporter colors the axis labels character-by-character. Do not use a
global string replacement pass for this: it corrupts the inserted
`<span class="axis-x">...` markup by recoloring the `x/y/z` letters inside the
HTML attributes themselves.

For the prototype join sketch:

```bash
python3 ../prototype_poormans_join.py --animate --axis-gizmo
```

The current debug rhythm is intentionally slower than the first pass:

- hold the exact canonical frame for about 1 second;
- move for 72 frames at 20 FPS, about 3.6 seconds;
- hold the final descendant for about 1.4 seconds.

The animation also blends numerically from the exact canonical terminal matrix
into the 3D-lit renderer during the first part of motion. That avoids the
visible renderer snap where the canonical mark was previously replaced by a
different-looking neutral 3D shape before the rotation became readable.

The debug profiles deliberately scale the final camera pose beyond the first
candidate search result. Current `clean` diagnostics can reach roughly 70
degrees of yaw, and the camera interpolation uses a sharper surge curve so the
middle of the motion is visibly faster than the beginning and end.

Current no-snap diagnostic exports:

```bash
xdg-open examples/debug_axis_surface_nosnap_big/index.html
xdg-open examples/debug_axis_contour_nosnap_big/index.html
xdg-open examples/debug_axis_ribbon_nosnap_big/index.html
```

### Playback quality decisions

- Every animation is precomputed before playback to avoid render-time stutter.
- Frame zero is exactly the canonical renderer output.
- The last frame is exactly the selected deterministic descendant.
- Continuous parameters use smooth eased interpolation.
- Depth starts first, camera motion begins later, and deformation/detail appears after that. This reads better than constant linear motion because the viewer can understand the object before it turns.
- The `lit` renderer uses approximate z-buffering, screen-space Lambert shading, subtle specular response, and depth fog. This is the main reason the result reads as 3D rather than as a flat density remix.
- Terminal output is emitted in one write per frame to reduce tearing.
- Playback uses the alternate screen and hides/restores the cursor safely.
- Frame pacing uses a monotonic clock instead of accumulating `sleep()` drift.
- The renderer automatically shrinks to fit the active terminal unless `--no-auto-fit` is used.

### Performance guidance

This engine searches for a deterministic target and precomputes every frame
before playback. On the current local machine:

- full `clean` profile: about 4.3 seconds to precompute;
- full `hero` contour-stack profile: about 4.7 seconds to precompute;
- cheap `clean --candidates 1 --frames 24`: about 0.9 seconds to precompute.

For a real `poormans join` flow, use one of these approaches:

- precompute/cache the canonical-to-machine animation once;
- use `clean --candidates 1 --frames 24` for cold startup;
- show the static canonical mark immediately and compute the animation while package work runs.

The full `hero` profile is best for demonstrations, exports, and ceremonial
screens. It should not block routine status commands.

### Tests

```bash
python3 -m unittest discover -s tests -v
```

The test suite checks exact canonical/final endpoints, deterministic reproduction
of a code-specific target, and exact default-canonical equality with the pinned
selected De Jong mark when that asset is present in the checkout.

## Native 3D object-first search

`generate_native_identity.py` is the drawing-board reset. It does not extrude
the De Jong mark or derive a delayed Z coordinate from it. It integrates genuine
three-dimensional continuous-time systems with RK4, rejects insufficiently
volumetric orbits, then searches camera orientations for a strong terminal logo.

Initial families:

- cyclic 3D De Jong
- cyclic 3D Clifford
- symmetric cyclic 3D trigonometric map
- 3D Lissajous
- two-harmonic 3D Fourier/Lissajous
- Aizawa
- Thomas
- Halvorsen
- Dadras
- Arneodo

Object rejection runs before projection scoring. It measures covariance
eigenvalue ratios, multiscale voxel occupancy/box dimension, and octant
occupancy. It also measures covariance inside spatial neighborhoods. That local
test is necessary because a curled sheet or tube can span all three global axes
while remaining locally flat. A family is allowed to fail; the search manifest records rejected
objects and the reason instead of silently weakening the volumetric threshold.

The registry distinguishes `volume` and `curve` topology. Cyclic De Jong,
Clifford, trigonometric maps, and the continuous attractors must pass the local
volume test. Lissajous and Fourier-Lissajous objects must instead be globally
non-planar and occupy 3D space; they are intentionally locally one-dimensional.
The gallery displays this classification so a knot is never presented as a
solid or volumetric attractor.

Run a quick search:

```bash
python3 generate_native_identity.py \
  --out examples/native3d_search \
  --per-family 3 \
  --points 14000 \
  --cameras 48

xdg-open examples/native3d_search/index.html
```

Run a broader final search:

```bash
python3 generate_native_identity.py \
  --out examples/native3d_search_full \
  --per-family 8 \
  --points 22000 \
  --cameras 96 \
  --keep 32 \
  --max-per-family 5
```

The browser viewer holds on the selected canonical camera, rotates to a
deterministic target, then holds again. The point cloud is immutable throughout:
there is no depth growth, layer separation, topology switch, or bitmap morph.
The exported `manifest.json` records the full object and projection scores, and
each candidate also receives a standalone metadata JSON file and terminal text
render.

Current local taste-test galleries:

```bash
# De Jong, Clifford, symmetric trig, Lissajous, Fourier, and native ODE families
xdg-open examples/native3d_variations/index.html

# Deeper sampling of Aizawa, Halvorsen, and Arneodo
xdg-open examples/native3d_existing_variations/index.html
```

Gallery camera workbench:

- drag to orbit yaw/pitch;
- hold Shift while dragging to roll;
- use the mouse wheel to zoom;
- switch to `Terminal` to rasterize the current camera live at 60x30;
- use `Save view` to preserve the terminal text and exact camera metadata;
- restore, copy, or download snapshots as TXT or JSON from `Snapshots`.

Selecting `Terminal` or manually moving the camera pauses automatic playback so
the animation cannot overwrite an inspected pose. Direct terminal links are
also supported with `?candidate=1&mode=terminal&still=1`.

## Transient laboratory

`generate_transient_lab.py` builds a separate browser laboratory for formation
animations. The De Jong reference video uses an ensemble transient: a dense
domain of initial conditions is pushed through the map together, once per
visual step. It is not a single orbit accumulating points.

```bash
python3 generate_transient_lab.py
xdg-open examples/transient_lab/index.html
```

The lab supports:

- classic and selected-poormans 2D De Jong;
- 2D Clifford;
- cyclic 3D De Jong, Clifford, and symmetric trigonometric maps;
- Aizawa, Thomas, Halvorsen, Dadras, and Arneodo RK4 ensemble flows;
- explicitly labeled Lissajous and Fourier parametric constructions;
- line, sheet, or cloud initial ensembles;
- 20k, 50k, or 100k particles;
- play, pause, single-step, restart, speed, exposure, and auto-fit controls;
- mouse camera orbit for 3D systems;
- PNG capture and in-browser WebM recording.

Parametric curves do not possess transient/recurrent attractor states, so the
lab does not mislabel their animation. It gradually activates their defining
curve from a line and identifies the protocol as `parametric construction`.

Deterministic inspection links accept `system`, `seed`, `density`, `step`, and
`still` query parameters. For example:

```text
examples/transient_lab/index.html?system=dejong2d_classic&step=8&still=1
```
