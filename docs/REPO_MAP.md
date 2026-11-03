# Repository map

What every file does, the order in which the pipeline runs, and where each number quoted in the README comes from. Nothing in `results/`, `models/`, `animations/` or `data/` was changed or regenerated when this map was written; the notebooks' output cells are the executed record and were not re-run.

## Pipeline order

```
data/airfrans_4digit_coefficients.csv ──► notebooks/03 (surrogate) ──► models/03/*.keras, *.joblib
                                                                    │
data/airfrans_cl_cd_surrogate.keras + scalers (the *provided* surrogate) ──► notebooks/04 (flap controller in the surrogate)
notebooks/01 (KS equation: the method) ──► notebooks/02 (thin-airfoil gust control) ──► notebooks/04
```

1. **Stage 1** `notebooks/01_ks_control_gymnasium.ipynb` establishes the method on the forced Kuramoto–Sivashinsky equation: spectral ETDRK4 solver, Gymnasium environment, local LQR baseline, PPO agent, eight experiments.
2. **Stage 2** `notebooks/02_airfoil_gust_control_thin_airfoil.ipynb` applies it to a 2D airfoil with thin-airfoil lift (2πα) and a flap, trained with TD3; seven experiments.
3. **Stage 3** `notebooks/03_airfrans_cfd_surrogate.ipynb` trains an MLP surrogate (Re, α, m, p, t) → (CL, CD) on the AirfRANS coefficient table; seven experiments (one not run, see below).
4. **Stage 4** `notebooks/04_airfoil_gust_control_cfd_surrogate.ipynb` replaces the lift law by the surrogate of `data/` and trains the flap controller with SAC inside it; seven experiments.

Stages 1 and 2 are independent of Stage 3. Stage 4 does **not** use the surrogate retrained by Stage 3 (`models/03_airfrans_cfd_surrogate/`); it uses the files in `data/`, which are an earlier training of the same architecture on the same table (compared in the last cell of Stage 3).

## Files

### Top level

| File | Role |
| --- | --- |
| `README.md` | Project summary, setup, data, headline results, reproduction commands, limitations. |
| `requirements.txt` | Frozen macOS environment (Python 3.12) in which the notebooks were executed. Contains macOS-only packages (`appnope`, `pyobjc-*`). |
| `requirements-test.txt` | Linux/CPU subset of the same pins, enough to import `src/airfoil_drl` and run `tests/`. |
| `pyproject.toml` | Makes `src/airfoil_drl` installable (`pip install -e .`) and configures pytest. |
| `.github/workflows/tests.yml` | CI: Python 3.12 on `ubuntu-latest`, CPU, runs `pytest` and `scripts/evaluate_pretrained.py` on every push and pull request. |
| `.gitignore` | Ignores checkpoints, caches and the training snapshot pickles (`animations/*.pkl`, which are not in the repository). |

### `src/airfoil_drl/` — code extracted from the notebooks

The notebooks originally defined these inline; they now import them (a `sys.path` line in each notebook's import cell points at `../src`, so no installation is needed to run them from `notebooks/`). The code is moved verbatim; only `AirfoilSurrogateGymEnv` gained an explicit `aero_fn` argument instead of reading a notebook global, and Stage 4 binds it in a two-line subclass.

| Module | Contents | Origin |
| --- | --- | --- |
| `ks.py` | `etdrk4_coefficients`, `periodic_distance`, `KSSystem` (ETDRK4 solver parameterised by domain length and actuators, `attractor()` generator), `reduced_linear_model`, `design_lqr`, `twin_divergence`, `lyapunov_slope`, `make_mpc`, `summarise` | Stage 1, Sections 3–4 and the experiments' shared setup |
| `thin_airfoil.py` | `one_minus_cosine_gust`, `static_lift_coefficient`, `AirfoilGymEnv`, `ZeroPolicy`, `run_episode`, `evaluate_on_gust`, `fly`, `suite_metrics`, `TEST_CASES`, `SUITE` | Stage 2, Sections 4–6, 7, 10–12 and the experiments' shared setup |
| `surrogate.py` | `geometry_parameters_from_name`, `is_four_digit_case`, `FEATURES`, `TARGETS`, `build_surrogate_model` (64-64-32 tanh MLP, Adam 1e-3, MSE), `score`, `dense_layers`, `mlp_forward`, `make_numpy_surrogate`, `load_numpy_surrogate` | Stage 3 Part A, Section 14, experiments; Stage 4 Section 4 (NumPy forward pass) |
| `surrogate_env.py` | `low_order_lift`, `low_order_aero`, `AirfoilSurrogateGymEnv(aero_fn, ...)` | Stage 4, Sections 6 and 19 |
| `metrics.py` | `episode_metrics` (CL RMSE, flap use, mean CD, return), `regression_metrics` (MAE, RMSE, R²) | Stages 2 and 4 Section 11; Stage 3 Section 16 |
| `geometry.py` | `naca4_outline`, `rotate` | flight animations of Stages 2 and 4 |

What stayed in the notebooks: anything that reads notebook globals (`KSGymEnv`, `KSExerciseEnv`, `evaluate`, `train_ppo`, `train_agent`, `make_agent`, the snapshot callbacks, the dashboards, Stage 4's `fly`/`suite_metrics` variant with the trusted-angle columns, the Keras `predict_airfrans_aero`, `fixed_gust_rollout`, `airfoil_shape`) and all training, plotting and file-writing cells.

### `notebooks/`

| File | Role |
| --- | --- |
| `0N_*.ipynb` | The four stages, with outputs. All paths inside are relative to `notebooks/`. |
| `0N_*.html` | HTML exports with outputs for reading without a kernel. They were exported **before** the extraction into `src/`, so their code cells show the inline definitions as they were executed; the `.ipynb` sources now import the same code. |

Two quirks of the executed record, left as they are because fixing them would require re-execution:

- Some output cells print the folder names of an earlier layout (`trained_models/`, `exercise_results/`, `animations/surrogate_training_dashboard.mp4`): the paths in the code were later renamed to `models/`, `results/` and `animations/03_*` without re-running those cells. The markdown of Stages 1 and 2 ("saved under `trained_models/exercises/`…") has the same old names.
- Stage 1 was executed with its PPO agent **loaded** from `models/01_ks_control_gymnasium/ppo/` (cell 22 output), so no training-snapshot pickle was written in that run; the animation cells read an existing pickle that is not in the repository.

### `scripts/`

| File | Role |
| --- | --- |
| `evaluate_pretrained.py` | Re-scores the saved agents and the provided surrogate on CPU in about 15 s, no training: Stage 1 cost over 20 unseen chaotic states, Stage 2 and 4 fixed-gust lift RMSE, Stage 3 test RMSE. `--stage N` selects stages. No scripts existed before this file; the notebooks are the pipeline. |

### `tests/`

| File | Covers |
| --- | --- |
| `test_ks.py` | ETDRK4 coefficient limits, actuator shapes, unstable-mode count against `results/01/ex1_domain_length.csv`, zero-mean invariant, attractor reproducibility, LQR closed-loop spectral radius 0.8143 (Stage 1 Section 7 output), Lyapunov fit on a synthetic exponential, MPC bounds. |
| `test_thin_airfoil.py` | Gust shape, lift laws, `check_env`, 120-step deterministic episode, uncontrolled metrics equal to the Stage 2 Section 11 output, `suite_metrics` equal to the "uncontrolled" row of `q1_algorithm_comparison.csv`. |
| `test_metrics_geometry.py` | Metrics on hand-computed cases, NACA 0012/2412 outline thickness and symmetry, rotation. |
| `test_surrogate.py` | Case-name parsing, table shape (489 = 378 + 111), MLP parameter count 6,690 and a 2-sample forward pass on CPU, NumPy forward pass equal to Keras, scaler round trip, provided surrogate equal to `provided_surrogate_evaluation.csv`. |
| `test_surrogate_env.py` | `AirfoilSurrogateGymEnv` with a cheap aero law (`check_env`, target lift, first-order flap, parameters reach the aero law), uncontrolled episode with the provided surrogate equal to the Stage 4 Section 11 table. |
| `test_pretrained_agents.py` | Saved TD3 (Stage 2) and SAC (Stage 4) agents reproduce their notebook RMSE; saved PPO (Stage 1) beats no control on the regenerated attractor. |
| `test_scripts.py` | Every script's argument parser builds and parses its flags. |

### `data/`

| File | Role |
| --- | --- |
| `airfrans_4digit_coefficients.csv` | 489 four-digit NACA simulations from AirfRANS (378 train / 111 test): `Re`, `Re_M`, `alpha_deg`, `m`, `p`, `t`, `CL`, `CD`, `split`. Built by Part A of Stage 3 from the full dataset (switched off; the notebook reads this file). |
| `airfrans_cl_cd_surrogate.keras`, `airfrans_cl_cd_scalers.joblib` | The provided surrogate (6,690 parameters) and its `StandardScaler`s; the aerodynamic law of Stage 4 and of the companion repository. |

### `models/`

| Path | Role |
| --- | --- |
| `01_ks_control_gymnasium/ppo/ks_ppo_final.zip` + `monitor.csv` | Main Stage 1 agent (PPO, 100,000 steps; the monitor log records 54 s of training). |
| `01_ks_control_gymnasium/came_with_the_notebook/{ppo,sac,td3}/` | Three agents trained elsewhere, scored without training in Appendix B of Stage 1. |
| `01_ks_control_gymnasium/exercises/` | PPO agents of experiments 2, 3, 4 and 6, with monitor logs. |
| `02_airfoil_gust_control_thin_airfoil/airfoil_0d_td3_final.zip` | Main Stage 2 agent (TD3, 50,000 steps). |
| `02_airfoil_gust_control_thin_airfoil/exercises/` | Agents of questions 1–6 and the nonlinear-lift extension, with monitor logs. |
| `03_airfrans_cfd_surrogate/airfrans_cl_cd_surrogate.keras` + scalers | Surrogate retrained by the Stage 3 run (not the one Stage 4 uses). |
| `03_airfrans_cfd_surrogate/exercises/ensemble_member_{0..4}` | Five members of the experiment-7 ensemble. |
| `04_airfoil_gust_control_cfd_surrogate/airfoil_surrogate_sac_final.zip` | Main Stage 4 agent (SAC, 50,000 steps). |
| `04_airfoil_gust_control_cfd_surrogate/exercises/` | Agents of experiments 1, 2, 5, 6, 7, with monitor logs. |

The notebooks reload the experiment agents (and the Stage 1 main agent) from these files when present; the main agents of Stages 2 and 4 and the Stage 3 surrogate are retrained on every run.

### `results/`

One CSV (and sometimes a PNG) per experiment, written by the experiment cells; the file names match the experiment numbers in the notebooks.

| Folder | Files |
| --- | --- |
| `01_ks_control_gymnasium/` | `ex1_domain_length` (csv, png), `ex2_controllability.csv`, `ex2_fewer_actuators` (csv, png), `ex3_sensor_limited.csv`, `ex4_threshold_reward.csv`, `ex5_saturation_failures.csv`, `ex6_robustness_length` (csv, png), `ex7_mpc_comparison.csv`, `ex8_lyapunov` (csv, png), `pretrained_models_scored.csv` (Appendix B). |
| `02_airfoil_gust_control_thin_airfoil/` | `q1_algorithm_comparison.csv`, `q2_no_gust_observation` (csv, png), `q3_smoothness_penalty` (csv, png), `q4_slow_flap.csv`, `q5_weak_training_gusts.csv`, `q6_drag_weight.csv`, `ext_nonlinear_lift.csv`. |
| `03_airfrans_cfd_surrogate/` | `ex1_architectures.csv`, `ex2_without_reynolds.csv`, `ex3_single_output.csv`, `ex4_scarce_data` (csv, png), `ex5_extrapolation.csv`, `ex6_five_digit_NOT_RUN.txt` (needs the 9 GB dataset), `ex7_ensemble.csv`, `ex7_ensemble_polar.png`, `provided_surrogate_evaluation.csv`. |
| `04_airfoil_gust_control_cfd_surrogate/` | `ex1_algorithm_comparison.csv`, `ex2_model_fidelity` (csv, png), `ex3_reynolds_sensitivity` (csv, png), `ex4_geometry_sensitivity.csv`, `ex5_reward_sensitivity.csv`, `ex6_domain_protection.csv`, `ex7_robust_reynolds` (csv, png). |

### `animations/`

Training dashboards (`*_training_dashboard.mp4` and `_final_frame.png`) for all four stages, the KS field evolution (`01_ks_field_evolution.mp4`) and the airfoil flight through the gust for Stages 2 and 4 (`*_airfoil_flight.mp4`, `*_flight_last_frame.png`). They are rendered from training-snapshot pickles that are written during training and are not in the repository.

## Where the README numbers come from

Cell numbers are 0-based indices into the `.ipynb` cell list (code and markdown cells together), as shown by `nbformat`/`json`.

| README statement | Source |
| --- | --- |
| 489 simulations, 378 train / 111 test, 5 inputs, 2 outputs | Stage 3 cell 19 and cell 25 outputs; `data/airfrans_4digit_coefficients.csv` |
| Stage 1: cost 114.33 (no control), 275.52 (local LQR), 14.63 ± 2.54 (PPO) over 20 unseen chaotic states | Stage 1 cell 25 output; identical in `results/01_ks_control_gymnasium/pretrained_models_scored.csv`; reproduced by `scripts/evaluate_pretrained.py --stage 1` |
| Stage 1: 240-step cost 32.4 and final energy 0.076 (PPO) versus 377.5 / 0.85 (no control) | Stage 1 cell 29 output; `pretrained_models_scored.csv` |
| Stage 1: SAC and TD3 agents trained elsewhere reach costs 4.78 and 5.43 | `results/01_ks_control_gymnasium/pretrained_models_scored.csv`; Stage 1 cell 61 |
| Stage 1: nonlinear MPC cost 25.8 at 0.26 s per decision versus PPO 14.7 at 0.3 ms | `results/01_ks_control_gymnasium/ex7_mpc_comparison.csv`; Stage 1 cell 54 |
| Stage 2: lift RMSE 0.1385 uncontrolled, 0.0166 with TD3, flap at most 7.96° on the fixed gust | Stage 2 cell 29 output; reproduced by `tests/test_thin_airfoil.py` and `tests/test_pretrained_agents.py` |
| Stage 2: mean RMSE over the 5-gust suite 0.151 → 0.0165 (TD3), 0.0083 (PPO), 0.0257 (SAC) | `results/02_airfoil_gust_control_thin_airfoil/q1_algorithm_comparison.csv`; Stage 2 cell 46 |
| Stage 3: test RMSE CL 0.0209 (R² 0.9986), CD 0.0010 (R² 0.963) for the surrogate retrained in the notebook; thin-airfoil CL RMSE 0.360 | Stage 3 cells 40 and 49 outputs; row "retrained in this execution" of `results/03_airfrans_cfd_surrogate/provided_surrogate_evaluation.csv` |
| Stage 3: the provided surrogate used by Stage 4 has CL RMSE 0.0259 (R² 0.998), CD RMSE 0.00087 (R² 0.972) | row "provided (used by Stage 4)" of `provided_surrogate_evaluation.csv`; Stage 3 cell 91; reproduced by `tests/test_surrogate.py` |
| Stage 3: 6,690 parameters, early stopping after 364 epochs (best at 324) | Stage 3 cells 29, 33 and 34 outputs |
| Stage 3: with 38 training cases CL RMSE 0.19, with 189 it is 0.032 | `results/03_airfrans_cfd_surrogate/ex4_scarce_data.csv` |
| Stage 4: target CL 0.7595; lift RMSE 0.1354 uncontrolled, 0.0596 with SAC; flap at most 9.6° | Stage 4 cells 21 and 38 outputs; reproduced by `tests/test_surrogate_env.py` and `tests/test_pretrained_agents.py` |
| Stage 4: PPO 0.0098 and TD3 0.0188 mean suite RMSE against SAC 0.0542 | `results/04_airfoil_gust_control_cfd_surrogate/ex1_algorithm_comparison.csv`; Stage 4 cell 69 |
| Stage 4: the SAC agent trained on the surrogate and tested on the thin-airfoil law has mean RMSE 0.144, about the uncontrolled 0.151 | `results/04_airfoil_gust_control_cfd_surrogate/ex2_model_fidelity.csv` |
| Stage 4: equivalent angle stays within 2.5°–10.6°, inside the training range of the CFD data (−4.9° to 14.8°) | `ex1_algorithm_comparison.csv` columns `alpha_eq_min/max`, `fraction_outside_trusted`; Stage 3 cell 21 output for the data range |
| Training wall times (Apple M3 Max): PPO 100k steps ≈ 55 s; TD3 50k ≈ 130–165 s; SAC 50k ≈ 165–180 s; PPO 50k ≈ 10 s; one surrogate fit ≈ 10 s | last `t` column of the `monitor.csv` files in `models/*/exercises/` and `models/01_ks_control_gymnasium/ppo/monitor.csv`; Stage 3 cell 70 output ("30 s for 3 seeds") |
| Tests and re-scoring run in about 15 s each on CPU | measured in the cloud session that added them (37 tests, 10–14 s; script 10 s) |
