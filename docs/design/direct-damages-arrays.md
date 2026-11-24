# Design: array interface for Monte Carlo direct damages

Status: implemented as `snail.damages.simulate_direct_damages` (draw scheme
`philox4x64-keyed-v1`), with the decisions recorded [below](#decisions).
`calculate_direct_damages` is unchanged.

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
    intensity,  # (split, event) for ONE scenario combination
    asset_index,  # (split,) parent asset of each split, 0..n_asset-1
    quantity,  # (split,) in the units of the cost
    curves=[curve_a, curve_b],
    curve_weights=[0.5, 0.5],
    cost_mean=2.0e6,  # scalar, or (asset,)
    cost_systemic_cv=0.2,
    cost_asset_cv=0.3,
    return_periods=[10, 100, 1000],
    quantiles=[0.05, 0.5, 0.95],
    n_samples=10_000,
    seed=20261114,
    asset_keys=stable_ids,  # (asset,) int64, keys the random streams
    keep_total_samples=True,
)
result.split_fraction  # (split, event, quantile)
result.split_cost  # (split, event, quantile)
result.split_ead  # (split, quantile)
result.asset_cost  # (asset, event, quantile)
result.asset_ead  # (asset, quantile)
result.total_cost  # (event, quantile)
result.total_ead  # (quantile,)
result.total_samples  # (sample, event), to sum across calls
```

One call per asset type and scenario combination ("case") gives everything the
workflow needs: no all-ones tricks, no repeated curve evaluation, and split,
asset and total results that come from the same samples.

## Function

```python
def simulate_direct_damages(
    intensity: ArrayLike,  # (n_split, n_event)
    asset_index: ArrayLike,  # (n_split,) int
    quantity: ArrayLike,  # (n_split,)
    *,
    curves: Sequence[DamageCurve],
    curve_weights: Sequence[float],
    cost_mean: float | ArrayLike,  # scalar or (n_asset,)
    cost_systemic_cv: float = 0.0,
    cost_asset_cv: float = 0.0,
    hazard_cv: float | ArrayLike = 0.0,  # scalar or (n_event,)
    eta_sd: float = 0.0,
    n_samples: int = 1000,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    return_periods: ArrayLike | None = None,  # (n_event,) years; enables EAD
    seed: int = 0,
    asset_keys: ArrayLike | None = None,  # (n_asset,) int64 or uint64; default arange
    systemic_key: int | None = None,  # keys curve choice and systemic cost draws
    n_assets: int | None = None,  # default asset_index.max() + 1
    levels: Collection[str] = ("split", "asset", "total"),
    keep_total_samples: bool = False,
    block_size: int | None = None,  # target splits per block; None: ~2**22 / (n * e)
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
    quantiles: numpy.ndarray  # (q,)
    return_periods: numpy.ndarray | None  # (e,)
    split_fraction: numpy.ndarray | None  # (s, e, q)
    split_cost: numpy.ndarray | None  # (s, e, q)
    split_ead: numpy.ndarray | None  # (s, q)
    asset_cost: numpy.ndarray | None  # (a, e, q)
    asset_ead: numpy.ndarray | None  # (a, q)
    total_cost: numpy.ndarray | None  # (e, q)
    total_ead: numpy.ndarray | None  # (q,)
    total_samples: numpy.ndarray | None  # (n, e)
    metadata: dict[str, Any]  # seed, inputs, curve weights, version of the draw scheme
```

`levels` selects which arrays are computed. For example, `("asset", "total")`
skips the split arrays entirely. Split and asset arrays are marginal quantiles
of the same samples; a split's samples sum to its asset's samples, sample by
sample.

Two public helpers summarise samples that have been added up across calls:

```python
summarise_samples(samples, quantiles)  # (..., n) -> (..., q)
expected_annual_damage(samples, return_periods)  # (n, e) -> (n,)
```

`expected_annual_damage` is the existing `_integrate_ead_samples`, made public.

## Random draws keyed by asset

The key change: every random variate depends only on `(seed, stream, key)`,
not on the order of the call.

| Draw | Stream | Key | Shape |
|---|---|---|---|
| damage curve choice | `curve` | `systemic_key` (optional) | (n,) |
| systemic cost factor | `system` | `systemic_key` (optional) | (n,) |
| hazard factor | `hazard` | asset key | (n,) per asset |
| damage noise | `damage` | asset key | (n,) per asset |
| asset cost factor | `cost` | asset key | (n,) per asset |

Implementation (`philox4x64-keyed-v1`, recorded as `metadata["draw_scheme"]`):
each stream is a `numpy.random.Philox` (Philox4x64) generator with the 128-bit
key `(seed, stream id)`, stream ids 1 to 5 in the order of the table. Rather
than advancing one stream to `asset_key * n_samples`, which overflows for
64-bit hash keys, the asset key goes into the 256-bit counter:

| Streams | Starting counter |
|---|---|
| hazard, damage, cost | `(0, 0, asset_key, 0)` |
| curve, system, no `systemic_key` | `(0, 0, 0, 1)` |
| curve, system, with `systemic_key` | `(0, 0, systemic_key, 2)` |

Each row is `random_raw(n_samples)` from its own generator, so it does not
depend on any other row, and the first `m` samples are the same for any
`n_samples >= m`. Each uniform uses one 64-bit word, `((x >> 12) + 0.5) / 2**52`,
strictly inside (0, 1) so that its inverse normal CDF (`scipy.special.ndtri`)
is finite. Curve choice is the inverse CDF of the normalised weights on the
`curve` stream's uniforms, so a zero-weight curve is never chosen. Signed asset
keys (for example hashes) map to their two's complement bits as `uint64`;
`uint64` keys (as from `pandas.util.hash_array`) are used as they are. Keys
must be unique within a call.

Making the streams costs about 0.07 ms per asset per stream at 10,000 samples
(one `Philox` per asset), and the inverse CDF about 0.24 ms, so about 0.3 ms
per asset per stream in all. Only streams with non-zero uncertainty are drawn.

Consequences:

- **Results do not depend on how a caller divides the work.** Chunks of
  assets, parallel workers, or skipping dry assets all give the same values for
  every asset that is computed.
- **Totals add up across calls.** Total samples from separate calls (chunks,
  asset types, layers) with the same seed and `systemic_key` share curve
  choice and systemic cost draws at each sample index, so summing
  `total_samples` and then summarising gives correct quantiles of the combined
  total. This gives mu-star the layer totals it cannot compute now.
- **Systemic draws can be made independent.** With `systemic_key=None` the
  curve choice and systemic cost draws depend only on the seed, so they are
  shared by every call (perfectly correlated across asset types, layers and
  hazards). With an integer key they also depend on the key: for example, a
  different key per asset type gives independent systemic draws per type.
  Totals from calls with different keys can still be summed; they are then
  independent rather than perfectly correlated. Which is right is the
  caller's modelling choice. The key does not affect local draws.
- **Scenarios use common random numbers.** The same asset gets the same draws
  in every scenario combination, so differences between scenarios reflect the
  hazard, not sampling noise. Today each (asset, scenario) group gets its own
  local draws.
- **Asset keys must be stable.** Callers pass stable integer keys, for example
  a 64-bit hash of the asset id (`pandas.util.hash_array`). The default,
  `arange(n_asset)`, is stable only for a fixed asset order.

## Computing each quantity once

Per block of splits (sized to bound memory at about
`n_samples × block_size × n_event` doubles). `block_size` is a target number of
splits; blocks hold whole assets. The default, `None`, picks about
`2**22 / (n_samples × n_event)` splits (32 MB per block array). A fixed 4096,
as first proposed, would be about 3 GB per array at 10,000 samples and 9
events. Results do not depend on the block size, except for the floating-point
summation order of totals.

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
6. EAD at each level: integrate over events, per sample. The integration is
   linear in the event damages, so split and asset EAD use one weight per event
   (a matrix product), which equals `expected_annual_damage` up to rounding.
   Total EAD is `expected_annual_damage(total_samples)`.
7. Quantiles, by one sort per array. A full sort plus numpy's interpolation is
   bit-identical to `numpy.quantile`, and was about 3× faster at n = 10,000 in
   the mu-star benchmark. With no hazard or damage-noise uncertainty, split
   fraction samples take one of the curve values, so their quantiles come
   exactly from the curve choice counts, without sorting.

Internally, samples are ordered by chosen curve, so each curve applies to one
contiguous slice; only `total_samples` is returned, in the original sample
order.

There is no Python loop per group. Each block is a handful of array operations,
and assets never straddle blocks, so asset sums stay within a block.

Damage fractions carry no cost uncertainty because they are taken before step
3, which removes the need for unit quantities and costs.

## Skipping dry assets

If every curve gives zero damage at zero intensity, splits with zero intensity
in all events have zero damage in every sample. With keyed draws, the function
leaves them out without affecting anyone else: it returns exact zeros for them,
and for assets with no other splits, and makes no draws for such assets. Split
and asset events with zero intensity are also not sorted. Most flood runs have
a few percent of splits exposed, so this is the main saving. If any curve is
non-zero at zero intensity, nothing is skipped (`metadata["skip_dry"]`).

## Performance

Measured on a 4-core cloud container (numpy 2.5, Python 3.12): 10,000 splits of
2,000 assets, 9 events, 10,000 samples, 2 curves, EAD and 3 quantiles, and
either cost uncertainty only (`cost_systemic_cv`, `cost_asset_cv`) or full
uncertainty (also `hazard_cv`, `eta_sd`). Peak memory is the process's peak
resident set, of which about 220 MB is the imported libraries.

| Function | Exposed splits | Uncertainty | Levels | Time | Peak RSS |
|---|---|---|---|---|---|
| `calculate_direct_damages` | 5% | cost | asset + total | 31.5 s | 250 MB |
| `calculate_direct_damages` | 5% | full | asset + total | 32.6 s | 250 MB |
| `calculate_direct_damages` | 100% | cost | asset + total | 36.8 s | 250 MB |
| `simulate_direct_damages` | 5% | cost | split + asset + total | 1.2 s | 312 MB |
| `simulate_direct_damages` | 5% | full | split + asset + total | 2.5 s | 387 MB |
| `simulate_direct_damages` | 100% | cost | split + asset + total | 11.9 s | 335 MB |
| `simulate_direct_damages` | 100% | full | split + asset + total | 35.5 s | 383 MB |

The old function computes one level per call, and the mu-star workaround makes
three calls. At 100% exposure the new function's time is mostly sorting for
quantiles (one sort of 10,000 samples per split, asset and event), and, with
hazard uncertainty, evaluating curves on every sample.

## Relation to `calculate_direct_damages`

Keep `calculate_direct_damages` and its results unchanged for now. Later, it
can become a thin wrapper: group by scenario, call `simulate_direct_damages`
once per scenario, and build the same tables. That changes its random streams,
so it needs a release note and a version flag in the metadata. Tests then check
the two agree in distribution, not sample by sample.

## Decisions

1. **Common random numbers across scenarios: yes.** Local draws are keyed only
   by `(seed, stream, asset key)`, so an asset gets the same draws in every
   call and scenario combination.
2. **Systemic draws can be independent.** `systemic_key: int | None = None`.
   None keys the curve choice and systemic cost streams by `(seed, stream)`
   only, shared by every call with the same seed. An integer also keys them,
   so a caller passing a different key per asset type gets independent
   systemic draws per type. Total samples from calls with different keys can
   still be summed; they are then independent, not perfectly correlated. That
   is the caller's modelling choice.
3. **No mean.** Only quantiles are returned. Callers who need a mean can keep
   `total_samples`.
4. **Exact quantiles without sampling.** Not a separate API. Internally, split
   fraction quantiles with no hazard or damage-noise uncertainty are computed
   exactly from the curve choice counts, and equal those of the samples.
5. **Name.** `simulate_direct_damages`, returning `DirectDamageArrays`.
   `calculate_direct_damages` and its results are unchanged.

## Tests

In `tests/test_simulate_damages.py`:

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
- Results do not depend on `block_size` (1, 7, large).
- Common random numbers across scenarios, and `systemic_key` behaviour.
- Quantiles equal `numpy.quantile` of the samples to 1e-12.

Not written: a test that memory stays flat as the number of splits grows at a
fixed `block_size` (it would be slow and platform-dependent); the benchmark
above covers it.
