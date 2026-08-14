# De Jong Attractor Reconstruction

## Purpose

This directory records an attempt to reverse engineer the short De Jong
attractor animation from the Reddit post ["The De Jong Attractor, animation I
made that shows the transient states and the recurrent attractor
states"](https://www.reddit.com/r/math/comments/kle3yb/).

The source clip supplied for the investigation was:

```text
/home/t/Downloads/c7faupq8ct761.mp4
```

It is a 1920x1080, 10 FPS, 129-frame (12.9 second) video. Its extracted frames
are committed under [`dejong_original_frames/`](dejong_original_frames/).

The visual target is not just the final strange attractor. The important part
is the transient sequence: a filled sheet becomes a blade, bends into a
crescent, then folds and intersects until it reaches the recurrent attractor.

## Mathematical Model

The assumed Peter de Jong map is evaluated synchronously:

```text
x[n+1] = sin(a * y[n]) - cos(b * x[n])
y[n+1] = sin(c * x[n]) - cos(d * y[n])
```

The coefficient tuple proposed during the first reconstruction was:

```text
a = -2
b = -2
c = -1.2
d = 2
```

These coefficients remain plausible. They reproduce important transient
shapes and the general recurrent topology once the initial-condition problem
is corrected. They have not yet been proven to be the source's exact tuple.

## What The First Reconstruction Did

The Claude baseline is preserved as
[`dejong_reconstruction_claude.html`](dejong_reconstruction_claude.html). The
working reconstruction is [`dejong_reconstruction.html`](dejong_reconstruction.html).

The baseline initialized 70,000 points in this rectangle:

```js
x = (random() - 0.5) * 1.0; // [-0.5, 0.5]
y = (random() - 0.5) * 0.7; // [-0.35, 0.35]
```

It used a fixed viewport and rendered one complete mathematical iteration per
displayed state. Those states are committed under
[`dejong_ours_frames/`](dejong_ours_frames/).

The synchronized viewer
[`dejong_original_frames.html`](dejong_original_frames.html) shows the original
on the left and this baseline on the right.

## What Was Wrong

The initial rectangle was much too large to qualify as "close to zero." It
already covered a region where the trigonometric map has substantial
curvature. Consequently:

- baseline iteration 1 was already strongly curved;
- baseline iteration 3 already contained folds and intersections;
- by iteration 7 the baseline had largely skipped the coherent sheet phase.

The original behaves differently. Its first several generations remain filled,
coherent sheets. It develops the blade and crescent before becoming strongly
self-intersecting around frames 13-17.

This is mathematically consistent with a microscopic two-dimensional seed.
For small `z`:

```text
sin(z) ~= z
cos(z) ~= 1 - z^2 / 2
```

Therefore the image of a sufficiently small square is initially governed by
the map's local linear behavior. It remains a deformed sheet. Repeated
expansion eventually makes the set large enough to experience the nonlinear
parts of sine and cosine, at which point the folds and intersections appear.

An initial width near `1.0` encounters that nonlinearity immediately. An
initial width near `0.02` delays it for approximately the number of generations
seen in the source.

## Why A Microscopic Seed Still Fills The Video

The source appears to fit or normalize every generation independently before
rendering it. Its early microscopic sheet and its final attractor occupy a
similar fraction of the frame despite having very different mathematical
bounds.

This per-frame camera fit is important. With a fixed mathematical viewport, a
true near-zero seed would initially look like a dot. With per-frame fitting,
the tiny seed and every subsequent image fill the canvas, exposing the transient
geometry.

The likely source pipeline is therefore:

1. Generate many points in a very small rectangle or square around `(0, 0)`.
2. Apply exactly one synchronous De Jong transformation to every point.
3. Compute that generation's bounds.
4. Scale and center that generation to fit the output frame.
5. Render the current points only.
6. Repeat.

## The Final Comparison

[`dejong_tiny_seed_matrix.jpg`](dejong_tiny_seed_matrix.jpg) is the last and most
useful diagnostic image from the conversation.

- Rows compare selected generations: 0, 1, 2, 3, 4, 5, 7, 10, 13, 17, and 24.
- The left column contains the corresponding source frames.
- The remaining columns use the same proposed coefficients with square seed
  half-widths `0.1`, `0.01`, `0.001`, and `0.0001`.
- Every generated cell is independently fitted to its panel.

The `epsilon = 0.01` column provides the closest overall timing in this test.
It remains sheet-like in the early generations, reaches a crescent near
generation 10, and becomes complex near generation 17. This strongly supports
seed scale plus per-frame fitting as the primary explanation.

Smaller seeds do not change the eventual map; they delay when a visible region
becomes large enough to fold. Extremely small seeds therefore remain nearly
linear for too many generations.

## Ideas Tested And Rejected

### In-between animation frames

Adding interpolated positions between generation `n` and `n+1` can make UI
playback less abrupt, but it does not explain the source shapes. The discrepancy
was already present in the discrete mathematical states. The source's smooth
early sheets are not merely crossfades or tween frames.

### Point blur and antialiasing

Gaussian splatting, blur, point density, and color affect edge quality. They do
not turn a multiply folded state into a coherent sheet. Rendering was not the
primary topology problem.

### Sequential update order

Using the newly computed `x[n+1]` while calculating `y[n+1]` produces a
different dynamical system and did not reproduce the source sequence. The
synchronous equations above remain the working model.

### A broader starting distribution

Testing broad seeds made the early complexity worse. The evidence points in
the opposite direction: the source seed is genuinely close to zero.

## Current Best Read

Confidence-ranked conclusions:

1. **High confidence:** the baseline's starting rectangle is far too large.
2. **High confidence:** the source uses a per-generation camera fit or equivalent
   normalization.
3. **High confidence:** interpolation is optional presentation polish, not the
   mathematical solution.
4. **Medium confidence:** a seed half-width around `0.01` is near the correct
   order of magnitude.
5. **Medium confidence:** `(-2, -2, -1.2, 2)` is close to, and may be exactly,
   the source coefficient tuple.
6. **Unresolved:** exact seed aspect ratio, random distribution, orientation,
   camera padding, point count, density transfer, and coefficient precision.

## Recommended Next Reconstruction

Start from the Claude baseline, but make these changes before tuning colors or
playback:

```text
seed x,y near [-0.01, 0.01]
synchronous map update
one displayed state per true iteration
independent bounds and camera fit for every state
deterministic random seed for comparisons
white points on black while matching geometry
```

Then compare generations 1, 2, 3, 5, 7, 10, 13, and 17 against the extracted
source frames. First optimize seed scale and aspect ratio. Only after the
transient timing matches should the search vary `a`, `b`, `c`, and `d` to match
the precise silhouette and recurrent topology.

