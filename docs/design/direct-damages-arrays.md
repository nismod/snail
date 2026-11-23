# Design: array interface for Monte Carlo direct damages

Status: proposal, for discussion. Nothing here is implemented yet.

## Why

`calculate_direct_damages` takes a long table of splits and returns quantile
tables for one level of aggregation, chosen by the `asset_id` column. A caller
that wants results at several levels has to call it several times. The mu-star
damage workflow ([nismod/mu-star#42](https://github.com/nismod/mu-star/pull/42))
makes three calls per asset type, with the same inputs each time:

1. `asset_id` set to a split id, `quantity = cost_mean = 1` and no cost
   uncertainty, to get split damage fractions
2. `asset_id` set to a split id, to get split costs and expected annual damage
   (EAD)
3. `asset_id` set to the asset id, with `aggregate_assets=True`, to get asset
   costs, EAD and totals

Review on that PR called this "quite an ugly workaround" and asked whether it
calls for a different function signature. It does. The workaround:

- **recomputes everything three times**: the same damage curves are evaluated
  on the same intensities in each call, and the damage fraction is computed
  twice more than needed
- **is inconsistent**: local draws (hazard, damage noise, asset cost) are
  made per `asset_id` group, so splits and assets get different draws. Split
  and asset results agree in distribution but not sample by sample
- **cannot be split up**: draws depend on the order of groups in the call, so
  chunking, parallelising or skipping assets changes everyone's results
- **cannot be summed across calls**: only quantiles come back, so a layer total
  across asset types (each with its own curves and costs) is impossible
- **is slow and memory-hungry**: a Python loop per group (about 1.6 ms each),
  results built as lists of dicts (about 600 B per row), and every scenario in
  one call. Benchmarks for mu-star: 29 min for 100,000 road splits at 5%
  exposure, 12 scenario combinations and 10,000 samples; a 500,000-split layer
  does not fit in 8 GB

## Proposal in brief

Add one array-based function that draws once and summarises at every level
from the same samples:

```python
result = simulate_direct_damages(
    intensity,      # (split, event) for ONE scenario combination
    asset_index,    # (split,) parent asset of each split, 0..n_asset-1
    quantity,       # (split,) in the units of the cost
    curves=[curve_a, curve_b],
    curve_weights=[0.5, 0.5],
    cost_mean=2.0e6,          # scalar, or (asset,)
    cost_systemic_cv=0.2,
    cost_asset_cv=0.3,
    return_periods=[10, 100, 1000],
    quantiles=[0.05, 0.5, 0.95],
    n_samples=10_000,
    seed=20261114,
    asset_keys=stable_ids,    # (asset,) int64, keys the random streams
    keep_total_samples=True,
)
result.split_fraction  # (split, event, quantile)
result.split_cost      # (split, event, quantile)
result.split_ead       # (split, quantile)
result.asset_cost      # (asset, event, quantile)
result.asset_ead       # (asset, quantile)
result.total_cost      # (event, quantile)
result.total_ead       # (quantile,)
result.total_samples   # (sample, event), to sum across calls
```

One call per asset type and scenario combination ("case") gives everything the
workflow needs: no all-ones tricks, no repeated curve evaluation, and split,
asset and total results that come from the same samples.

## Function

```python
def simulate_direct_damages(
    intensity: ArrayLike,                 # (n_split, n_event)
    asset_index: ArrayLike,               # (n_split,) int
    quantity: ArrayLike,                  # (n_split,)
    *,
    curves: Sequence[DamageCurve],
    curve_weights: Sequence[float],
    cost_mean: float | ArrayLike,         # scalar or (n_asset,)
    cost_systemic_cv: float = 0.0,
    cost_asset_cv: float = 0.0,
    hazard_cv: float | ArrayLike = 0.0,   # scalar or (n_event,)
    eta_sd: float = 0.0,
    n_samples: int = 1000,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    return_periods: ArrayLike | None = None,   # (n_event,) years; enables EAD
    seed: int = 0,
    asset_keys: ArrayLike | None = None,  # (n_asset,) int64; default arange
    n_assets: int | None = None,          # default asset_index.max() + 1
    levels: Collection[str] = ("split", "asset", "total"),
    keep_total_samples: bool = False,
    block_size: int = 4096,               # splits per vectorised block
) -> DirectDamageArrays: ...
```

Inputs are plain arrays, aligned by position. The caller (mu-star, or the
existing DataFrame function) owns ids, labels and scenario dimensions. The
function knows only splits, the assets they belong to, events and samples.

`intensity` is for a single scenario combination: its events are, for example,
the return periods of one climate scenario and epoch. Callers loop over cases.
NaN intensity is an error. The caller decides whether missing data means zero
hazard or a missing case, as mu-star already does.

### Result

```python
@dataclass(frozen=True)
class DirectDamageArrays:
    quantiles: numpy.ndarray                   # (q,)
    return_periods: numpy.ndarray | None       # (e,)
    split_fraction: numpy.ndarray | None       # (s, e, q)
    split_cost: numpy.ndarray | None           # (s, e, q)
    split_ead: numpy.ndarray | None            # (s, q)
    asset_cost: numpy.ndarray | None           # (a, e, q)
    asset_ead: numpy.ndarray | None            # (a, q)
    total_cost: numpy.ndarray | None           # (e, q)
    total_ead: numpy.ndarray | None            # (q,)
    total_samples: numpy.ndarray | None        # (n, e)
    metadata: dict[str, Any]                   # seed, inputs, curve weights, version of the draw scheme
```

`levels` selects which arrays are computed. For example, `("asset", "total")`
skips the split arrays entirely. Split and asset arrays are marginal quantiles
of the same samples; a split's samples sum to its asset's samples, sample by
sample.

Two public helpers summarise samples that have been added up across calls:

```python
summarise_samples(samples, quantiles)                 # (..., n) -> (..., q)
expected_annual_damage(samples, return_periods)       # (n, e) -> (n,)
```

`expected_annual_damage` is the existing `_integrate_ead_samples`, made public.

## Random draws keyed by asset

The key change: every random variate depends only on `(seed, stream, key)`,
not on the order of the call.

| Draw | Stream | Key | Shape |
|---|---|---|---|
| damage curve choice | `curve` | none | (n,) |
| systemic cost factor | `system` | none | (n,) |
| hazard factor | `hazard` | asset key | (n,) per asset |
| damage noise | `damage` | asset key | (n,) per asset |
| asset cost factor | `cost` | asset key | (n,) per asset |

Implementation: a counter-based generator (`numpy.random.Philox`) keyed by
`(seed, stream)` and advanced to `asset_key * n_samples`. Normals come by
inverse CDF from `Generator.random`, which consumes exactly one 64-bit word
per double, so each asset's block sits at a fixed position in its stream. A
prototype confirmed that drawing for a subset of assets, in any order, gives
bit-identical values to drawing for all of them. It costs about 0.3 ms per
asset per stream at 10,000 samples, against about 0.2 ms for ordinary
sequential draws.

Consequences:

- **Results do not depend on how a caller divides the work.** Chunks of
  assets, parallel workers, or skipping dry assets all give the same values for
  every asset that is computed.
- **Totals add up across calls.** Total samples from separate calls (chunks,
  asset types, layers) share curve choice and systemic cost draws at each
  sample index, so summing `total_samples` and then summarising gives correct
  quantiles of the combined total. This gives mu-star the layer totals it
  cannot compute now.
- **Scenarios use common random numbers.** The same asset gets the same draws
  in every scenario combination, so differences between scenarios reflect the
  hazard, not sampling noise. Today each (asset, scenario) group gets its own
  local draws.
- **Asset keys must be stable.** Callers pass stable integer keys, for example
  a 64-bit hash of the asset id (`pandas.util.hash_array`). The default,
  `arange(n_asset)`, is stable only for a fixed asset order.

## Computing each quantity once

Per block of splits (sized to bound memory at about
`n_samples × block_size × n_event` doubles):

1. Hazard samples: `h = max(0, intensity × (1 + hazard_cv × z_hazard[asset]))`.
2. Damage fraction samples: evaluate each curve on the intensities it is
   chosen for, then apply noise and clip:
   `f = clip(curve[choice](h) × exp(eta_sd × z_damage[asset] − eta_sd²/2), 0, 1)`.
   With `hazard_cv = eta_sd = 0` (the usual case), `h` is the same in every
   sample. Each curve is then evaluated once per split and event, and `f` is a
   gather by curve choice.
3. Split cost samples: `c = f × quantity × unit_cost[asset]`, where
   `unit_cost = max(0, cost_mean × systemic × (1 + cost_asset_cv × z_cost[asset]))`.
4. Asset cost samples: sum `c` over each asset's splits.
5. Total samples: sum over assets.
6. EAD at each level: integrate over events, per sample.
7. Quantiles, by one sort per array. A full sort plus numpy's interpolation is
   bit-identical to `numpy.quantile`, and was about 3× faster at n = 10,000 in
   the mu-star benchmark.

There is no Python loop per group. Each block is a handful of array operations,
and assets never straddle blocks, so asset sums stay within a block.

Damage fractions carry no cost uncertainty because they are taken before step
3, which removes the need for unit quantities and costs.

## Skipping dry assets

If every curve gives zero damage at zero intensity, assets with zero intensity
in all events have zero damage in every sample. With keyed draws, the function
can leave them out without affecting anyone else. It returns exact zeros for
them and makes no draws. Most flood runs have a few percent of splits exposed,
so this is the main saving.

## Expected performance

From the mu-star benchmark of a vectorised prototype, before keyed draws: 4–10×
faster per group than the current loop, and 9.9× end to end on real roads
(1,740 s to 175 s) when combined with computing damage fractions once.
Memory is bounded by `block_size`, not by the number of splits. These numbers
need re-measuring with keyed draws.

## Relation to `calculate_direct_damages`

Keep `calculate_direct_damages` and its results unchanged for now. Later, it
can become a thin wrapper: group by scenario, call `simulate_direct_damages`
once per scenario, and build the same tables. That changes its random streams,
so it needs a release note and a version flag in the metadata. Tests then check
the two agree in distribution, not sample by sample.

## Open questions

1. **Common random numbers across scenarios.** Is it right that an asset gets
   the same local draws in every scenario? It helps comparisons, but it is a
   change from today.
2. **Systemic draws across asset types.** Keyed by `(seed, stream)` only, so
   they are shared by every asset type, layer and hazard, which is mu-star's
   documented choice. Should a caller be able to pass a key to make them
   independent per asset type?
3. **Quantiles of damage fraction with several curves and no other
   uncertainty** are exact values from one curve or another, and the median
   can jump with the seed. Return the mean too?
4. **Exact quantiles without sampling.** With no continuous uncertainty, split
   fraction quantiles follow from the curve weights alone. Worth a fast path,
   or is the sorted-sample route quick enough?
5. **Name.** `simulate_direct_damages`, or reuse `calculate_direct_damages` for
   the array form and rename the table form?

## Tests to write

- Split samples sum to asset samples, and asset samples to total samples,
  sample by sample.
- Draws for a subset of assets, in any order, or in separate calls, are
  bit-identical to drawing all at once.
- Summed `total_samples` from two calls equal the total from one call over
  both sets of assets.
- With no uncertainty, every quantile equals the deterministic value. EAD
  matches an independent implementation.
- Skipped dry assets give exact zeros and leave other assets unchanged.
- Agreement in distribution with `calculate_direct_damages` for the same inputs.
- Memory stays flat as the number of splits grows at a fixed `block_size`.
