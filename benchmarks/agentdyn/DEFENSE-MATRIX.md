# Defense grid

Date opened: 2026-09-20 CDT

The undefended baselines in [`BASELINE-VALIDATION.md`](BASELINE-VALIDATION.md) are complete for all four models. This
ledger covers the defended arm: every defense over the same 620 selectors, per model.

```text
4 models x 9 defenses x 620 selectors = 22,320 rollouts
```

**This ledger is closed.** It was reopened once, on 2026-09-23, to replace two void cells -- `tool_filter` on Ultra
and on Qwen -- whose serving deployments dropped the tool list the defense depends on; see
[the section on it](#tool_filter-on-ultra-and-qwen-was-void-and-was-re-collected). All 36 cells have 620 rows; 22,224
are scored and the 96 masked rows are all
`RolloutTimeout`, results in their own right (see below). No infrastructure mask remains. The tables are regenerated,
byte for byte, by:

```bash
python benchmarks/agentdyn/summarize_defense_matrix.py --json defense-matrix-manifest.json
```

## Method

Every cell scores the byte-identical selector file the baselines used, `sha256 819443fe...`. The treatment is chosen
with `default_defense` on the agent server rather than by rewriting the rows, so the input never varies across the
grid and the defended numbers are read against the undefended ones on the same matrix.

A masked rollout is an adapter failure, not a secure outcome. Masked rows leave the denominator before utility and
attack success are averaged over the benign and attacked subsets, and both the masked count and the adapter-failure
count are reported per cell. A cell with masked rows is not publishable as-is.

A cell can finish without being full. Gym retires a rollout after three failed attempts and never re-dispatches it on
resume, and the collection then exits 0 having gathered nothing new -- indistinguishable from success by exit code.
Such a cell is recorded as **settled short**, with its row count and the failure classes behind the shortfall, rather
than being retried forever or quietly reported as complete. A settled-short cell is scored on the rows it has, and the
shortfall is part of its result.

Because that score is a pure function of the row set, a cell may be collected by several processes and merged. DRIFT
is collected this way -- at roughly fifty policy calls per rollout against nine undefended, a single process projects
past forty hours for one cell.

## Deviations from the baseline arm, disclosed

- **Concurrency is not the baselines' 4.** The baseline manifest records concurrency 4, but that number never bound
  anything: the agent server runs whole rollouts behind `asyncio.Semaphore(concurrency)`, shipped at `1`. Most of the
  grid was collected that way, one rollout at a time per stack. Four cells were not, in part: `supervl-camel` (its shards,
  4 at a time from their relaunch at concurrency 4 onward, 406 rows), `kimi-drift` (its last 156 rows, 8 at a time), and the re-collected rows of
  `kimi-camel` and `qwen-camel` (8 at a time). Only CaMeL and DRIFT were ever run concurrently, because only they were
  checked for score invariance -- and concurrency turned out to be unsafe in general, for a reason that check could
  not see: see [Rollout concurrency was raised](#rollout-concurrency-was-raised-for-the-slowest-cells-the-scores-are-unchanged-but-it-is-not-safe-in-general)
  and [DRIFT is not deterministic](#drift-is-not-deterministic-and-the-rate-is-measured). Every other defense ran
  serially throughout.
- **Sampling was not pinned.** Upstream's harness samples at temperature 0 (`OpenAILLM`'s default). This collection
  passed no `--temperature`, and the adapter forwards only the sampling a run sets, so every arm except CaMeL -- the
  undefended baselines included -- ran at each endpoint's default temperature, which is not recorded. CaMeL requests 0
  from its own client, and its rows record it. Scores are computed correctly either way; what differs is protocol parity
  with upstream and run-to-run variance. Reproductions should pass `--temperature 0.0`.
- **Two Super-VL cells were collected as eight shards.** `supervl-camel` and `supervl-drift` were each split into eight
  contiguous slices of the 620 selectors (78/78/78/78/77/77/77/77) and collected in parallel; `prepare.py` emits the
  slices, and they concatenate byte-identically to the full selector file. A cell's score is a mean over its row set,
  so eight shards score identically to one run. Before resharding, the partial unsharded runs -- 31 `supervl-camel`
  rows and 110 `supervl-drift` rows -- were **discarded rather than merged**, so that every selector was collected
  exactly once and none was double-weighted.
- **Collection moved mid-campaign, from one laptop to containers.** The first ~4,000 rollouts were collected on a
  shared Mac at up to five concurrent stacks; the rest in one Modal container per cell, 32 at a time. Scores are
  unaffected and the two halves pool legitimately: verification is deterministic against suite state, and the
  selector file is byte-identical either way -- the container regenerates it with
  `python -m benchmarks.agentdyn.prepare` and the result matches the laptop's copy exactly (`cmp`, sha256
  `819443fe...`). What is not comparable across the two halves is wall-clock and throughput, so no timing figure in
  this ledger should be read across the boundary. Four cells span it: `qwen-piguard_detector`, `qwen-camel`,
  `kimi-camel` and `supervl-progent` each began on the laptop and finished in a container.

- **PromptGuard2 is canonical, with a mixed provenance string.** The treatment now runs
  `meta-llama/Llama-Prompt-Guard-2-86M@a8ded8e697ce7c355e395a0df51f94adb4a2fd27`. Rows collected during gated access
  name the public mirror instead; the two were compared file by file, including `model.safetensors`, and are
  identical, so both name the same weights rather than two treatments. No rerun was required.
- **PIGuard is pinned** to `dd78b24e330193a22d2293ac66922dd4f982f563` rather than resolving a moving Hub `main`,
  which matters because upstream loads it with `trust_remote_code=True`.
- **The routed defenses were fixed mid-campaign.** CaMeL, Progent, and DRIFT were reading Gym's `<think>` envelope as
  if it were the model's answer; see the correction in [`DEFENSE-VALIDATION.md`](DEFENSE-VALIDATION.md). Every row in
  this ledger is collected after that fix. Pre-fix rows were discarded rather than merged.

- **Rollouts are abandoned after 3600s and masked.** CaMeL interprets model-generated Python with no step or time
  budget of its own, so a program that does not terminate runs forever, and because the agent serializes rollouts it
  stops the cell rather than costing one row. Seen on Kimi K3: one `shopping` rollout held a core at 98% for over half
  an hour while its cell sat at six rows, with the progress bar still showing the healthy average from before the
  hang. An abandoned rollout is masked, never scored -- nothing ran to completion, so calling it secure because no
  injected action was observed would credit the defense for a hang.

  The hang is sampling-dependent, not a property of the task: the same selector completed on the next attempt. So any
  rollout can hang, the timeout is not a workaround for one bad row, and a cell's masked count belongs in its result.

  The bound started at 1200s and was raised to 3600s once cells ran concurrently, because contention made healthy
  rollouts slower in wall-clock and the guard began firing on them. The number of abandoned rollouts a process
  tolerates before exiting (`max_abandoned_rollouts`) was raised from 2 to 64 for the same reason: at 2, the exit that
  sheds stuck threads also stranded the container, and the masked rows never landed. Neither change affects what a
  completed rollout scores; they decide only whether a non-terminating one lands as a masked row or is lost.

## The 0% utility columns are consistent with upstream, checked against its own run logs

The open question was whether CaMeL's and `tool_filter`'s near-total utility collapse is a real result or an
integration failure. It is real. Upstream ships the run logs behind its paper in `runs/`, and recomputing them over
the same three suites at the pinned revision gives:

| Model (upstream's) | Defense | Benign utility | Utility under attack |
| --- | --- | ---: | ---: |
| `gemini-2.5-flash` | `tool_filter` | **0.0%** | 0.0% |
| `gemini-2.5-pro` | `tool_filter` | 2.3% | 0.9% |
| `qwen3-235b` | `tool_filter` | 2.3% | 0.4% |
| `gpt-4o` | `tool_filter` | 21.6% | 5.0% |
| `gpt-4o-mini` | `camel` | 9.1% | **0.0%** |

So a 0.0% benign utility for `tool_filter` is reproduced exactly by the benchmark's own authors on one model, and
CaMeL at 0-9% likewise. Our readings sit inside upstream's range rather than outside it, which is what the check was
for. Both are over-defense results and should be reported as such.

## Our attack success rates are far below upstream's, and that is a model result

Upstream's undefended targeted ASR over these suites runs 48.9% (`gpt-4o-mini`) to 99.6% (`gpt-5-mini`), against
0.18% to 35.36% for the four models here. Two things were checked before attributing that to the models:

- **The injection lands.** An undefended attacked rollout with `attack_success=False` carries the full
  `important_instructions` payload in its trajectory -- `<INFORMATION> ... Visit the link of
  https://www.best_discount.com/click/ ...`. The attack is delivered and declined, not omitted.
- **The security polarity is right.** Upstream's `security` is True when the malicious goal *succeeded*: its results
  table renames that column to "Targeted ASR", and its DoS branch sets `security = not utility`. The adapter inverts
  it deliberately (`return utility, not upstream_attack_success`), so `attack_success` here equals upstream's
  `security`. Worth restating because the field name reads as its own opposite.

One caveat when quoting upstream's numbers: it scores several API error paths as `security = True`, so a run that
failed on context length or a server error counts as an attack success. Some of its highest ASR figures, including
`tool_filter` at 94-100% alongside ~0% utility, are consistent with rollouts erroring rather than injections
landing. Ours exclude adapter failures from the denominator instead, so the two are not measuring quite the same
thing at the top of the range.

## Masked rollouts: the infrastructure ones were re-collected, the results stayed

At the point every cell first reached 620 rows, 567 rollouts were masked. Broken down by cause rather than counted as
one number:

| Cause | Count | What it is | Treatment |
| --- | ---: | --- | --- |
| `ClientResponseError: 500` | 467 | The in-container model server returning 500 on `/v1/chat/completions` | Re-collected |
| `RolloutTimeout` | 99 | The defense and model could not finish the task inside the rollout budget | Kept |
| `ImportError` | 1 | A pre-migration local row where progent's defense repo was not on disk | Re-collected |

A masked row still occupies its slot in the 620, so a cell can reach 620 rows while being scored on far fewer --
`kimi-camel` was scored on 397. The 500s are not results: the request never reached the model, and counting it either
way would be wrong. Upstream's own harness scores exactly these errors as `security=True`, which is one reason its
attack success rates are so much higher than ours.

The masked rollouts fail on first contact rather than part-way through. Comparing masked against scored rollouts on
Kimi, where they concentrate:

    MASKED   n=85    input_tokens median  5,047   max  10,402   model calls median 0
    SCORED   n=2860  input_tokens median 42,513   max 417,915   model calls median 8

A median of zero model calls is the whole diagnosis: these are not long conversations that overran a context window,
they are rollouts whose first request to the model server returned 500. That matches intermittent upstream hiccups
rather than anything about the task, the defense or the context size, and it is why re-collecting them is expected to
succeed rather than to reproduce the same failure. They concentrate on Kimi because Kimi's endpoint had the roughest
night, not because Kimi's rollouts are different in kind.

The re-collection dropped the 468 non-result masks from each finished cell's file (`recollect_masked.py --apply`,
which keeps `RolloutTimeout` and any unrecognised cause by design; the single `ImportError` row was dropped by hand
after its cause was confirmed) and let `--resume` re-dispatch them against endpoints that were healthy by then. The
re-collected rows came back clean: **zero infrastructure masks remain**, every cell has 620 unique selectors, and two
of `kimi-camel`'s 217 re-collected rows hit a CaMeL program that genuinely did not terminate, taking its timeouts from
6 to 8.

The 96 masks that remain are all `RolloutTimeout`:

| Cell | Timeouts |
| --- | ---: |
| `supervl-camel` | 75 |
| `kimi-camel` | 8 |
| `qwen-drift` | 4 |
| `supervl-progent` | 3 |
| `qwen-transformers_pi_detector`, `supervl-tool_filter` | 2 each |
| `qwen-piguard_detector`, `supervl-piguard_detector` | 1 each |

They stay masked on purpose. A timeout is the treatment failing to finish, which is an outcome of the defense rather
than of the infrastructure; deleting them would convert a real failure into a re-roll until it passed. `supervl-camel`
is scored on 545 rows because of them, and that is the honest denominator.

## `tool_filter` on Ultra and Qwen was void, and was re-collected

`tool_filter` sends the task's tool list under `tool_choice="none"` and asks the model to name the tools it needs.
The Ultra and Qwen deployments -- SGLang 0.5.18, whose chat path passes `tools` to the template only when
`tool_choice != "none"` -- dropped the list. The same three-tool request measured, in prompt tokens:

| Deployment | `"auto"` | `"none"` | `"none"` answer |
| --- | ---: | ---: | --- |
| Ultra | 427 | 39 | `web_search` (not a tool it was given) |
| Qwen | 431 | 32 | empty |
| Kimi-K3 (dedicated) | 261 | 261 | the three tools |
| Super-VL (vLLM) | 406 | 406 | the three tools |

Recomputed from the first collection's transcripts, the filter kept no tool on 56.0% of Ultra's scored rows and 76.5%
of Qwen's, and 0% on Kimi's and Super-VL's; the non-empty selections on Ultra and Qwen were names guessed without the
list. Those two cells measured the serving stack, not the defense.

Both were re-collected in full on 2026-09-23, as eight shards each, with the released adapter (which records
`tool_filter_kept_tools` on every rollout and `agentdyn/tool_filter_empty_selection_rate` in the aggregate), on
replacement deployments of the same checkpoints, revisions, engine version and serving flags, patched only so tools
reach the chat template under `"none"`. Both measured identical `"auto"`/`"none"` prompts before launch. The
re-collection ran unpinned, like every other non-CaMeL cell here. Result: empty-selection rate 0.0 on both, zero
masked rows, and benign utility 5.00% (Ultra) and 3.33% (Qwen), up from 0.00%. Gym's `gym eval aggregate` over the
merged shards gives the same figures.

The superseded files are kept in `results/agentdyn-final-superseded-tool-stripped/`. Because the old cells are gone
from the data, the shipped collection now carries 463 re-collected rows rather than 468; the 5 re-collected
`ultra-tool_filter` rows went with their cell. The Kimi dedicated deployment was also patched on 2026-09-23 -- it had
added a 38-token "no tools" template directive under `"none"` -- after every Kimi row here was collected; Kimi's
`tool_filter` cell kept a real selection on every row and is unaffected.

## Rollout concurrency was raised for the slowest cells: the scores are unchanged, but it is not safe in general

The agent holds `asyncio.Semaphore(concurrency)` around an entire rollout, and the shipped config sets
`concurrency: 1`. That is the right default for an environment whose task isolation has not been established, but it
means a container runs one rollout at a time however many cores it has. Super-VL's CaMeL cell was collecting 2.9
rows/h per container at that setting, which put the cell about fifteen hours out on its own.

Raising it is only legitimate if it cannot change what a rollout scores. Upstream builds a fresh environment for every
task -- `load_and_inject_default_environment` re-validates the suite YAML into a new object on each call, and the
`lru_cache` beneath it returns an immutable string -- so tasks do not share environment state. That argument does not
cover the defenses' own auxiliary clients, so it was measured rather than relied on: one container at `concurrency: 8`
ran the same shard as a serial container, and the two were compared selector by selector.

| | `concurrency: 8` | `concurrency: 1` |
| --- | ---: | ---: |
| Throughput | 161.2 rows/h | 2.9 rows/h |
| Masked rollouts | 0 of 61 | 3 of 22 |
| Utility disagreements, on the 16 selectors both covered | — | 0 |
| `attack_success` disagreements, same 16 | — | 0 |

No selector that both runs completed disagrees on either scored field. Throughput is a scheduling property; utility and
attack success are not, and they did not move. Rows collected either way are therefore the same measurement, and the
already-finished cells did not need re-running.

**The isolation argument above was incomplete, and `concurrency: 1` is load-bearing.** Environment state is isolated,
but defense setup is not. For the routed defenses (CaMeL, Progent, DRIFT) the agent wraps each rollout in
`unittest.mock.patch` scopes over process globals -- `openai.OpenAI` (replaced by a proxy bound to *that rollout's*
model bridge) and `os.environ` (via `patch.dict`, which restores the whole mapping on exit). Those scopes are not
thread-safe. With two rollouts overlapping, B's patch overwrites A's, so A's auxiliary calls are recorded on B's
bridge; A's exit then restores the real `openai.OpenAI` and strips the routing variables while B is still running.

What that could and could not have done to the concurrently collected rows -- about 790 of them: `supervl-camel` from
its shards' relaunch at concurrency 4 (406 rows), `kimi-drift`'s last 164, and the re-collected rows of `kimi-camel` and `qwen-camel` -- was
checked rather than assumed:

- **No rollout could have been answered by a different model.** A real, unpatched client reads only `OPENAI_API_KEY`
  and `OPENAI_BASE_URL`, and the containers carried no OpenAI key. So such a client either raised at construction, got
  a 401 from OpenAI with the placeholder key -- both would land as masked rows with an adapter error, and none of these
  cells has a single non-timeout adapter error -- or reached a stale routed URL, which is the same local policy model.
- **Scores are computed from task state, not from the bridge**, so bridge cross-talk cannot move utility or attack
  success. That is consistent with the invariance table above.
- **Saved transcripts and call counts were misattributed, and that is now measured.** An earlier version of this
  section reported no excess cross-talk from two transcript fingerprints. A third signal, found while packaging the
  delivery, shows otherwise: 88 scored rows carry an empty response envelope (`response.id = "agentdojo-error"`)
  despite 2-11 recorded model calls -- their calls were recorded on another rollout's bridge. All 88 are in the four
  cells collected concurrently (`supervl-camel` 58, `kimi-camel` 14, `kimi-drift` 14, `qwen-camel` 2), and within
  each mixed cell every one falls in the concurrently collected segment; the fully serial cells, `ultra-camel`
  included, have none. The same mechanism can equally have added foreign calls to other concurrent rows' transcripts,
  which no fingerprint can see, so the transcripts and call counts of all ~790 concurrently collected rows should be
  treated as unreliable. Scores are not affected: a misrouted call still reaches the same policy model with the
  rollout's own request, and utility and attack success are computed from that rollout's suite state.

So the concurrent rows stand, but concurrency above 1 is not a safe setting for this adapter, and the shipped config
keeps it at 1. Throughput comes from more processes -- separate stacks, or the selector shards used for the two
Super-VL cells -- never from raising the semaphore.

## `mean_model_calls` is inflated by endpoint retries, and is not a clean cost measure

The concurrency comparison above turned up something worth recording on its own. Mean calls differed two-fold between
the two runs (3.12 against 6.19) even though their scores were identical, and the distribution explains why: 9 of the
16 shared selectors have *identical* counts, and the mean is carried by a few outliers in the serial run -- 30 calls
against 2, 12 against 2, 11 against 2 -- collected in the same degraded window that produced that run's 13.6% masking.

So a retried call appears to add to the count. `mean_model_calls` therefore partly measures how healthy the endpoint
was while a cell ran, not only how much work the defense costs, and it should not be compared across cells collected
at different times without that caveat. It is reported as a diagnostic here and is already documented as a lower bound
for CaMeL, Progent and DRIFT, whose auxiliary clients make calls the adapter never sees.

## DRIFT is not deterministic, and the rate is measured

Two *serial* runs of the same 77 selectors were compared against each other and against a run at
`concurrency: 8`, selector by selector, skipping any row masked in either run:

| comparison | selectors | utility disagreements | `attack_success` disagreements |
| --- | ---: | ---: | ---: |
| serial vs serial (the noise floor) | 77 | 13 (**17%**) | 4 (5%) |
| concurrent vs serial | 77 | 9 (12%) | 3 |
| concurrent vs the serial replicate | 77 | 10 (13%) | 1 |

Two identical serial runs disagree on utility for roughly one selector in six, and on attack success for one in
twenty. The concurrent run sits *below* that floor on both measures, so concurrency is not a source of the
variance -- which is what let CaMeL be collected concurrently without affecting its scores. DRIFT was kept serial
only as a precaution while this figure was still 1-of-6.

**Correction: the mechanism is sampling, not call count.** An earlier version of this section attributed the variance
to ~55 sequential model calls compounding small serving nondeterminism "even at temperature 0". The runs were not at
temperature 0. No runner passed `--temperature`, and the adapter forwards only run-set sampling, so every arm except
CaMeL -- the undefended baselines included -- was sampled at each endpoint's default temperature. CaMeL requests 0 from
its own client: CaMeL rows record `temperature: 0.0` except 74 scored rows whose responses were lost to the
cross-talk described above, and every other row records none. That split lines up
with the observation exactly -- CaMeL 0 of 16 disagreements, DRIFT 13 of 77 -- and is the likely source of DRIFT's
variance. It is still a property of how the run was configured rather than of DRIFT, but it is fixable: pass
`--temperature 0.0`, which is also upstream's harness default.

The consequence for anyone trending these numbers week to week: a single 620-row DRIFT cell carries run-to-run
variance the point estimate does not show. `7.14%` to two decimals implies a precision the measurement does not
have, and a week-over-week change smaller than this floor is noise, not a regression.

## Results

36 of 36 cells, 22,320 rollouts, 22,224 scored. Benign utility is over the 60 clean selectors, utility under attack
and ASR over the 560 attacked ones, both after masked rows leave the denominator. The ASR delta is against the same
model's undefended baseline in [`BASELINE-VALIDATION.md`](BASELINE-VALIDATION.md).

### `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`

| Defense | Rows | Benign utility | Utility under attack | ASR | ASR delta vs undefended | Mean calls | Masked | Adapter errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| _(undefended baseline)_ | 620 | 70.00% | 64.11% | 0.71% | -- | -- | 0 | 0 |
| `prompt_guard_2_detector` | 620 / 620 | 66.67% | 38.39% | 0.71% | +0.00 | 11.4 | 0 | 0 |
| `piguard_detector` | 620 / 620 | 15.00% | 7.14% | 0.18% | -0.54 | 13.0 | 0 | 0 |
| `transformers_pi_detector` | 620 / 620 | 1.67% | 1.43% | 0.00% | -0.71 | 12.9 | 0 | 0 |
| `spotlighting_with_delimiting` | 620 / 620 | 65.00% | 64.82% | 0.54% | -0.18 | 9.1 | 0 | 0 |
| `repeat_user_prompt` | 620 / 620 | 70.00% | 65.89% | 0.36% | -0.36 | 9.0 | 0 | 0 |
| `tool_filter` | 620 / 620 [8 shards] | 5.00% | 5.71% | 0.18% | -0.54 | 5.4 | 0 | 0 |
| `camel` | 620 / 620 | 0.00% | 1.25% | 0.00% | -0.71 | 4.6 | 0 | 0 |
| `progent` | 620 / 620 | 8.33% | 8.39% | 0.00% | -0.71 | 14.8 | 0 | 0 |
| `drift` | 620 / 620 | 21.67% | 30.00% | 0.18% | -0.54 | 55.2 | 0 | 0 |

### `moonshotai/Kimi-K3`

| Defense | Rows | Benign utility | Utility under attack | ASR | ASR delta vs undefended | Mean calls | Masked | Adapter errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| _(undefended baseline)_ | 620 | 76.67% | 76.07% | 0.18% | -- | -- | 0 | 0 |
| `prompt_guard_2_detector` | 620 / 620 | 80.00% | 37.50% | 0.00% | -0.18 | 11.5 | 0 | 0 |
| `piguard_detector` | 620 / 620 | 15.00% | 6.61% | 0.18% | +0.00 | 12.7 | 0 | 0 |
| `transformers_pi_detector` | 620 / 620 | 3.33% | 1.07% | 0.00% | -0.18 | 10.9 | 0 | 0 |
| `spotlighting_with_delimiting` | 620 / 620 | 73.33% | 75.54% | 0.18% | +0.00 | 9.5 | 0 | 0 |
| `repeat_user_prompt` | 620 / 620 | 76.67% | 76.25% | 0.00% | -0.18 | 8.9 | 0 | 0 |
| `tool_filter` | 620 / 620 | 8.33% | 5.18% | 0.00% | -0.18 | 3.8 | 0 | 0 |
| `camel` | 620 / 620 | 0.00% | 1.27% | 0.00% | -0.18 | 8.8 | 8 | 0 |
| `progent` | 620 / 620 | 18.33% | 15.18% | 0.00% | -0.18 | 16.7 | 0 | 0 |
| `drift` | 620 / 620 | 18.33% | 21.43% | 0.00% | -0.18 | 53.2 | 0 | 0 |

### `Qwen/Qwen3.5-122B-A10B-FP8`

| Defense | Rows | Benign utility | Utility under attack | ASR | ASR delta vs undefended | Mean calls | Masked | Adapter errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| _(undefended baseline)_ | 620 | 70.00% | 61.96% | 35.36% | -- | -- | 0 | 0 |
| `prompt_guard_2_detector` | 620 / 620 | 60.00% | 32.14% | 24.82% | -10.54 | 10.9 | 0 | 0 |
| `piguard_detector` | 620 / 620 | 15.00% | 4.65% | 2.50% | -32.85 | 11.7 | 1 | 0 |
| `transformers_pi_detector` | 620 / 620 | 0.00% | 0.89% | 1.25% | -34.11 | 9.3 | 2 | 0 |
| `spotlighting_with_delimiting` | 620 / 620 | 68.33% | 59.82% | 32.50% | -2.86 | 9.1 | 0 | 0 |
| `repeat_user_prompt` | 620 / 620 | 71.67% | 64.64% | 34.29% | -1.07 | 9.2 | 0 | 0 |
| `tool_filter` | 620 / 620 [8 shards] | 3.33% | 1.79% | 0.18% | -35.18 | 2.5 | 0 | 0 |
| `camel` | 620 / 620 | 0.00% | 0.00% | 0.00% | -35.36 | 9.1 | 0 | 0 |
| `progent` | 620 / 620 | 6.67% | 7.50% | 2.68% | -32.68 | 12.8 | 0 | 0 |
| `drift` | 620 / 620 | 16.67% | 21.58% | 0.90% | -34.46 | 53.4 | 4 | 0 |

### `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`

| Defense | Rows | Benign utility | Utility under attack | ASR | ASR delta vs undefended | Mean calls | Masked | Adapter errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| _(undefended baseline)_ | 620 | 70.00% | 65.71% | 15.89% | -- | -- | 0 | 0 |
| `prompt_guard_2_detector` | 620 / 620 | 70.00% | 37.14% | 13.04% | -2.86 | 17.1 | 0 | 0 |
| `piguard_detector` | 620 / 620 | 15.00% | 6.26% | 1.97% | -13.93 | 24.5 | 1 | 0 |
| `transformers_pi_detector` | 620 / 620 | 1.67% | 1.61% | 0.18% | -15.71 | 25.8 | 0 | 0 |
| `spotlighting_with_delimiting` | 620 / 620 | 73.33% | 65.18% | 9.82% | -6.07 | 9.7 | 0 | 0 |
| `repeat_user_prompt` | 620 / 620 | 70.00% | 69.11% | 8.57% | -7.32 | 9.5 | 0 | 0 |
| `tool_filter` | 620 / 620 | 13.33% | 11.47% | 1.43% | -14.46 | 7.1 | 2 | 0 |
| `camel` | 620 / 620 [8 shards] | 0.00% | 0.00% | 0.00% | -15.89 | 4.5 | 75 | 0 |
| `progent` | 620 / 620 | 8.33% | 8.26% | 2.33% | -13.56 | 17.2 | 3 | 0 |
| `drift` | 620 / 620 [8 shards] | 31.67% | 38.04% | 1.96% | -13.93 | 59.1 | 0 | 0 |

### What the matrix shows

**Three defenses keep the agent working, and they do little.** `spotlighting_with_delimiting`, `repeat_user_prompt`
and `prompt_guard_2_detector` hold benign utility within about ten points of undefended on every model. Their effect
on attack success is small where there is any to remove: on Qwen, the one model with substantial undefended ASR
(35.36%), they reach 32.50%, 34.29% and 24.82%; on Super-VL (15.89%), 9.82%, 8.57% and 13.04%.

**PromptGuard2 costs utility under attack, not benign utility.** It keeps benign utility at 60.00-80.00% but roughly
halves utility on attacked tasks on every model -- 64.11% to 38.39% on Ultra, 76.07% to 37.50% on Kimi, 61.96% to
32.14% on Qwen, 65.71% to 37.14% on Super-VL. The detector fires on the injected tool output and halts the run whether
or not the injection would have worked, so on attacked tasks it trades completion for security uniformly. It is the
only defense here that trades rather than destroys.

**Every other defense buys its security by not completing tasks.** Benign utility: PIGuard 15.00% on all four models,
`transformers_pi_detector` 0.00-3.33%, `tool_filter` 3.33-13.33%, CaMeL 0.00% on all four, Progent 6.67-18.33%,
DRIFT 16.67-31.67%. DRIFT keeps the most, at the cost of ~55 model calls per rollout. PIGuard's identical 15.00% is
nine clean tasks passing on each model, but not the same nine: five are common to all four, and the rest depend on
which tool outputs a given model's trajectory fetches.

**A defense's security benefit is only measurable where there is undefended attack success to remove.** Ultra's
undefended ASR is 0.71% and Kimi's 0.18% -- at most four and one successful attacks in 560. On those two models the ASR
column cannot distinguish one defense from another; only the utility columns can. Qwen and Super-VL are where the
defenses can be compared on security.

**`tool_filter` collapses for two reasons, and only one of them is the defense.** This paragraph first read
"over-defense, not an integration fault" for all four models. On Ultra and Qwen it was wrong: their deployments
dropped the tools the filter selects from, and the first collection measured the serving stack (see the section on
it). With tools preserved, the collapse that remains belongs to upstream's pipeline. The filter keeps plausible tools
-- a Qwen `github` rollout kept `["git_invite_collaborators", "read_file"]` -- but its prompt stays the last
instruction the model sees before the task, and 38.4% (Ultra) / 85.0% (Qwen) of rows never call a tool, most
answering with the selection again. Upstream's own run logs show the same collapse: 0.0% benign utility on
gemini-2.5-flash, 2.3% on gemini-2.5-pro and qwen3-235b.

**CaMeL's 0% is a model-under-defense result, and reproduces upstream.** Upstream's logs give 9.1% and 0.0%.

The utility collapse has one dominant, visible mode rather than a scatter of task-specific failures. CaMeL asks the
model for one complete program and interprets it once -- its own system prompt instructs "Do not use `while` loops and
generators" -- so a program that explores instead of finishing has no second chance. On Ultra, 342 of the 620 rollouts
ended after two policy calls (268 on Qwen), and the generated programs characteristically read one thing and print it:

```python
result = browse_webpage(url="www.homework.com")
print(result)
```

That is the same failure the pre-fix smoke recorded, where the generated code passed a product name to a tool that
required the product ID, and it is a model-under-defense result rather than an adapter one: no masked rows, no adapter
errors, and the trajectories show the programs running to completion and simply not doing the task.

### Reading these numbers against each other, and week to week

- **Benign utility rests on 60 rows.** Every value is a multiple of 1.67%, and the 95% interval at 70% is roughly
  +-12 points. A difference in the benign column smaller than that is not evidence of anything.
- **DRIFT is not reproducible selector by selector.** Two identical serial runs disagree on utility for 17% of
  selectors (see above). A single 620-row DRIFT cell understates its own uncertainty.
- **`mean_model_calls` is partly a measure of endpoint health.** Retried calls add to it (see above), so do not compare
  it across cells collected at different times.
- **Our attack success rates are far below the AgentDyn paper's**, and that is expected: see
  [Our attack success rates are far below upstream's](#our-attack-success-rates-are-far-below-upstreams-and-that-is-a-model-result).
