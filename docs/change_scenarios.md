# Change scenarios: how much code two routing changes needed

The review router (`src/roadcond/routing/`) was built so that a new signal or a new decision
policy plugs in without touching the router, the interfaces or the audit log. This page checks
that claim against two real changes, each made as its own commit on `feat/learned-policy`. Every
number below comes from `git diff` between the commits named, and can be re-checked with the
commands at the end.

| Scenario | Commit | Parent |
|---|---|---|
| (a) add a confidence provider (`MarginProvider`) | `4be0b09` | `e7b97e9` |
| (b) replace the decision policy (`LearnedPolicy`) | `944edc3` | `4be0b09` |
| not a scenario: fitting and comparison tooling | `7a645e1`, `fc550b8` | `944edc3` |

## (a) Add a confidence provider: `MarginProvider`

`git diff --stat e7b97e9 4be0b09`: **6 files changed, 168 insertions, 0 deletions**
(1 file added, 5 changed).

| File | Added | Removed | What |
|---|---:|---:|---|
| `src/roadcond/routing/providers.py` | 21 | 0 | the new class, appended after `CalibratedConfidenceProvider` |
| `src/roadcond/routing/run.py` | 9 | 0 | `PROVIDERS` registry and the `extra_providers` loop in `build_providers` |
| `src/roadcond/routing/__init__.py` | 2 | 0 | export |
| `configs/default.yaml` | 1 | 0 | `extra_providers: []` |
| `configs/routing_cnn.yaml` | 1 | 0 | `extra_providers: []` |
| `tests/test_learned_policy.py` (added) | 134 | 0 | tests |

Production code: 32 lines added in 3 files, none removed.

Lines changed where the design says none should be:

| File or interface | Lines changed |
|---|---:|
| `router.py` | 0 |
| `ReviewPolicy` Protocol (`policy.py`, whole file) | 0 |
| `ConfidenceProvider` Protocol (`providers.py`) | 0: the only hunk is `@@ -86,0 +87,21 @@`, an insertion after line 86; the Protocol block is byte-identical before and after |
| `log.py` | 0 |

Tests that prove it (9 functions, 13 collected tests; the suite went from 256 to 269 passed, 7
skipped):

- `test_margin_at_temperature_one_is_the_raw_top_two_gap`,
  `test_margin_uses_the_temperature_scaled_probabilities` (T = 0.5 and 2.0),
  `test_margin_is_zero_for_a_tie_and_one_for_a_certain_model`,
  `test_margin_refuses_a_bad_temperature`: the signal itself, at T = 1 and T ≠ 1.
- `test_invalid_margin_input_goes_to_the_routers_fallback` (4 bad inputs) and
  `test_a_margin_outside_zero_one_is_refused_by_the_router`: the router's existing fallback
  handles the new provider's failures, with no router change.
- `test_margin_never_sees_the_label_or_the_file_name`: it gets a `CaseView` only.
- `test_config_switches_the_margin_provider_on_by_name`: off by default, on by config, unknown
  names refused.
- `test_margin_is_logged_and_leaves_threshold_decisions_unchanged`: the signal appears in every
  decision line, and the threshold policy's modes and reasons are identical with and without it.

## (b) Replace the decision policy: `LearnedPolicy`

`git diff --stat 4be0b09 944edc3`: **5 files changed, 864 insertions, 1 deletion**
(1 file added, 4 changed).

| File | Added | Removed | What |
|---|---:|---:|---|
| `src/roadcond/routing/learned.py` (added) | 511 | 0 | the policy, the NumPy logistic regression, cutoff selection, the weights file |
| `src/roadcond/routing/run.py` | 5 | 0 | import, `"learned"` in `POLICIES`, and a 3-line guard: `--bundle-dir` refuses any policy but `threshold` |
| `configs/default.yaml` | 2 | 0 | `learned_weights`, `l2` |
| `configs/routing_cnn.yaml` | 2 | 0 | `learned_weights`, `l2` |
| `tests/test_learned_policy.py` | 344 | 1 | tests (the 1 removed line is the import list, rewritten) |

Production code outside the new module: 5 lines added in `run.py`, none removed.

| File or interface | Lines changed |
|---|---:|
| `router.py` | 0 |
| `ReviewPolicy` Protocol (`policy.py`, whole file) | 0 |
| `ConfidenceProvider` Protocol (`providers.py`, whole file) | 0 |
| `log.py` | 0 |

Why the guard in `run.py`: the live API (`live.py`) builds only the confidence provider and
routes with whatever `routing.json` names. Rather than change the API, the offline check refuses
to write a `routing.json` for any policy but `threshold`, so the live default cannot change by
accident.

Tests that prove it (21 functions, 28 collected tests; the suite went from 269 to 297 passed, 7
skipped):

- `test_the_learned_policy_satisfies_the_review_policy_protocol`: it plugs into the existing
  interface.
- `test_safety_class_wins_even_at_almost_zero_risk` (either model),
  `test_disagreement_wins_even_at_almost_zero_risk`,
  `test_a_missing_signal_gives_at_least_full_review` (each of the 3 signals),
  `test_an_unusable_prediction_gives_at_least_full_review`: the hard rules beat the model.
- `test_a_learned_policy_that_raises_falls_back_to_full_review`: the router's existing
  `fallback:policy_error`.
- `test_the_learned_policy_never_sees_the_label_or_the_file_name`.
- `test_the_weights_hash_is_in_the_version_and_every_logged_decision`,
  `test_edited_weights_get_a_new_version`, `test_a_missing_or_mismatched_weights_file_is_refused`:
  the version names the exact weights in every log line, through the unchanged `log.py`.
- `test_fit_logistic_recovers_known_coefficients_and_is_deterministic`,
  `test_l2_shrinks_the_coefficients_but_not_the_intercept`, `test_fit_logistic_refuses_bad_input`.
- `test_predicted_risk_picks_the_mode` (5 cases), `test_one_model_needs_no_agreement_signal`,
  `test_cutoff_for_count_comes_as_close_as_ties_allow`,
  `test_equal_workload_cutoffs_match_the_threshold_shares_to_one_image`,
  `test_decide_gives_the_modes_the_cutoffs_were_chosen_for`,
  `test_equal_catch_cut_full_is_the_lightest_cut_that_reaches_the_target`,
  `test_the_fit_refuses_any_split_but_val`.
- `test_the_live_api_is_offered_the_threshold_policy_only`.

## Not a scenario: the comparison tooling

`git diff --stat 944edc3 7a645e1`: 5 files changed, 738 insertions, 1 deletion
(`compare.py` 542 lines, two 14-line scripts, 9 lines in the `Makefile`, 159 lines of tests; the
1 removed line is the Makefile's `.PHONY` list). It measures the change; it is not needed to use
the new policy. Its 6 tests (303 passed, 7 skipped afterwards) include the end-to-end fit and
compare on mock data and the check that scrambling the test split changes neither the weights nor
any cutoff. A follow-up commit, `fc550b8` (2 files, 8 insertions), records the share of images the
hard rules alone send to second review; a docs commit then adds this page, the README section and
the guide.

## What the numbers do and do not show

- Neither change touched `router.py`, `policy.py`, `log.py`, `case.py`, `live.py`, the API or
  the frontend: 0 lines across the branch.
- Both changes still touched `run.py`, the one place that picks concrete providers and policies
  (a registry entry each), and the YAML configs.
- Adding keys to the `routing:` section changes `config_hash` for future runs of `make route` and
  `make route-cnn`, because the hash covers the whole section. Existing logs keep their old hash,
  and `serving/routing.json` still loads, because it carries its own routing section and hash.
- Line counts measure size, not difficulty. Most of scenario (b)'s 511 lines are the model and
  the cutoff selection, which any learned policy needs whatever the architecture.

## Re-checking

```bash
git diff --stat e7b97e9 4be0b09
git diff --stat 4be0b09 944edc3
git diff --numstat e7b97e9 944edc3 -- src/roadcond/routing/router.py \
    src/roadcond/routing/policy.py src/roadcond/routing/log.py      # prints nothing
git diff -U0 e7b97e9 4be0b09 -- src/roadcond/routing/providers.py | grep '^@@'
```
