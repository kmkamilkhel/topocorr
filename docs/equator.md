# Near the equator

Within about ten degrees of the equator, the sun passes north of a tile in part of the year and
south of it in the rest. Per-image c then differs between the two sides, and with Landsat 5's
drifting overpass the mix changes over the years. One curve c(solar zenith) cannot follow that:
on test tiles at 4 and 10 degrees latitude, `scsc-stack` turned a west-minus-east difference of
Landsat 5 NIR trends from positive to clearly negative (over-correction), where per-image SCS+C
left it near zero.

topocorr detects this case from the sun azimuths of the model's images:

```python
from topocorr.stack import sun_sides

model.sun_both_sides  # True when each side holds at least 5 % of the images
sun_sides(azimuths)  # (images with the sun north of the tile, images with it south)
```

When it is true:

- `correct(images, model)` warns and uses per-image SCS+C instead of `scsc-stack`, and per-image C
  instead of `c-stack`; `se-stack` (per-image slopes) is not affected;
- `diagnose` warns and records `tc_sun_both_sides = 1` on its output; its drift estimate rests on the
  one curve, so read it with care there;
- `compare` warns and records `tc_sun_both_sides = 1` in its table; it applies every method as named,
  so compare the stack methods with the per-image ones there.

The [Tapantí notebook](examples.md) (Costa Rica, 9.8° N) shows the case step by step.
