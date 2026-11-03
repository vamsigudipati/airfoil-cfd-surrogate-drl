# Airfoil gust control with a CFD surrogate and deep reinforcement learning

[![tests](https://github.com/vamsigudipati/airfoil-cfd-surrogate-drl/actions/workflows/tests.yml/badge.svg)](https://github.com/vamsigudipati/airfoil-cfd-surrogate-drl/actions/workflows/tests.yml)

This repository holds the first four stages of a project in which I take a flap controller for a two-dimensional airfoil from a textbook aerodynamic law to a neural-network surrogate built on RANS simulations. Stage 1 sets up the method on a chaotic field, the forced Kuramoto–Sivashinsky equation, wrapped as a Gymnasium environment in which PPO, SAC and TD3 agents from Stable-Baselines3 learn to suppress the chaos with a few actuators. Stage 2 applies the same method to gust rejection on a thin-airfoil model, where the lift follows 2πα and a flap is the control. Stage 3 replaces that law with a multilayer-perceptron surrogate trained on the AirfRANS coefficient table, mapping the Reynolds number, the angle of attack and the three NACA four-digit parameters to the lift and drag coefficients. Stage 4 puts the surrogate inside the control environment and trains the flap controller against it. Every notebook ends with the experiments and the extra experiments I ran, and each has a training animation. The second half of the project, reduced-order models of the flow fields and a design search in their latent space, is in the companion repository `airfoil-rom-betavae-design`.

Portfolio page: https://vamsigudipati.github.io/airfoil-cfd-surrogate-drl.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/01_ks_control_gymnasium.ipynb` | Stage 1 — Control of a chaotic field: the method | `results/01_ks_control_gymnasium/` | `models/01_ks_control_gymnasium/` | `animations/01_*` |
| `notebooks/02_airfoil_gust_control_thin_airfoil.ipynb` | Stage 2 — Gust control with a thin-airfoil model | `results/02_airfoil_gust_control_thin_airfoil/` | `models/02_airfoil_gust_control_thin_airfoil/` | `animations/02_*` |
| `notebooks/03_airfrans_cfd_surrogate.ipynb` | Stage 3 — A CFD surrogate from AirfRANS | `results/03_airfrans_cfd_surrogate/` | `models/03_airfrans_cfd_surrogate/` | `animations/03_*` |
| `notebooks/04_airfoil_gust_control_cfd_surrogate.ipynb` | Stage 4 — The flap controller trained inside the surrogate | `results/04_airfoil_gust_control_cfd_surrogate/` | `models/04_airfoil_gust_control_cfd_surrogate/` | `animations/04_*` |

Each notebook is also exported as HTML next to it, with all outputs, for reading without a kernel (the HTML shows the code as it was executed, before the helper functions moved to `src/`). The `results/` folders hold the tables (CSV) and figures (PNG) of the experiments; the `models/` folders hold the trained agents and surrogates (Stable-Baselines3 `.zip` files with their training logs, Keras `.keras` files with their scalers); `animations/` holds the training and flight videos (MP4) and their final frames. `docs/REPO_MAP.md` describes every file, the pipeline order and the source of every number below.

## Code layout

The pure functions the notebooks define (the ETDRK4 solver, the gust and lift laws, the two Gymnasium environments, the surrogate MLP and its NumPy forward pass, the metrics, the NACA geometry) live in the small package `src/airfoil_drl/`, from which the notebooks import them; the training, plotting and experiment cells stay in the notebooks. `tests/` checks the package against the notebook outputs and the saved agents, `scripts/evaluate_pretrained.py` re-scores the saved agents and the surrogate on CPU, and the GitHub Actions workflow runs both on every push.

## Headline results

One seed per configuration; all numbers are read from the notebook output cells and the CSV files named in `docs/REPO_MAP.md`.

| Stage | What is measured | Uncontrolled / baseline | Trained agent |
| --- | --- | --- | --- |
| 1 | Cost (mean field energy plus control effort) over 80 steps, averaged over 20 unseen chaotic states | no control 114.3, local LQR 275.5 | PPO 14.6 ± 2.5 |
| 1 | Same cost over 240 steps from one state, and the final energy | no control 377.5, energy 0.85 | PPO 32.4, energy 0.076 |
| 2 | Lift-coefficient RMSE through the fixed gust (2.5 m/s, 1.3 s), flap limited to ±20° | 0.1385 | TD3 0.0166 with at most 8.0° of flap |
| 2 | Mean lift RMSE over the 5-gust suite | 0.151 | TD3 0.0165, PPO 0.0083, SAC 0.0257 |
| 3 | Surrogate RMSE on the 111 held-out AirfRANS cases (lift / drag), 6,690 parameters | thin-airfoil 2πα: lift RMSE 0.360 | retrained in the notebook: 0.0209 (R² 0.999) / 0.0010 (R² 0.963); provided surrogate used by Stage 4: 0.0259 (R² 0.998) / 0.00087 (R² 0.972) |
| 4 | Lift RMSE through the fixed gust inside the surrogate (NACA 2412, Re 4×10⁶, target CL 0.7595) | 0.1354 | SAC 0.0596 with at most 9.6° of flap |
| 4 | Mean lift RMSE over the 5-gust suite | 0.150 | SAC 0.0542, TD3 0.0188, PPO 0.0098 |

Two findings worth stating plainly: in Stage 4 the main SAC agent is the weakest of the three algorithms on the suite (experiment 1), and the agents trained elsewhere for Stage 1 with SAC and TD3 reach costs of 4.8 and 5.4 against 14.6 for my PPO agent (Appendix B of Stage 1). The surrogate's equivalent angle of attack stays between 2.5° and 10.6° in the controlled episodes, inside the −4.9° to 14.8° range of the CFD training data.

## Setting up and running

I ran everything on a laptop CPU (Apple M3 Max) with Python 3.12. The frozen environment is in `requirements.txt`; the videos need `ffmpeg` on the path.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name airfoil-drl
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_ks_control_gymnasium.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there (or open them in Jupyter from that folder). The experiment agents and the Stage 1 agent are reloaded from `models/` when the files are present; the main agents of Stages 2 and 4 and the surrogate of Stage 3 are retrained by their notebooks (a few minutes each). The snapshot files that the animation cells read are written during training and are not in the repository; to rebuild the Stage 1 animation, move `models/01_ks_control_gymnasium/ppo/` aside so that the agent trains again. Stage 3 loads `data/airfrans_cl_cd_surrogate.keras` and `data/airfrans_cl_cd_scalers.joblib` as the reference surrogate in its last section, and Stage 4 uses the same two files as its aerodynamic law.

## Reproduce

Everything below runs on a CPU; `requirements-test.txt` is the Linux subset of the frozen environment (the full `requirements.txt` contains macOS-only packages). Wall times were measured on an Apple M3 Max laptop; a CI runner is about two to three times slower.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt && pip install -e .    # 2–5 min, downloads torch and tensorflow

python -m pytest -q                                          # 37 tests, about 15 s
python scripts/evaluate_pretrained.py                        # re-scores the saved agents and surrogate, about 15 s
```

The script prints the headline numbers next to the notebook values (Stage 1 costs 114.3 / 14.6, Stage 2 RMSE 0.1385 / 0.0166, Stage 3 RMSE 0.0259 / 0.00087, Stage 4 RMSE 0.1354 / 0.0596) and `tests/test_pretrained_agents.py` asserts them.

To re-execute the notebooks (from `notebooks/`, with the full environment and `ffmpeg`):

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_ks_control_gymnasium.ipynb   # ~5–10 min: agents reloaded; the MPC experiment dominates
jupyter nbconvert --to notebook --execute --inplace 02_airfoil_gust_control_thin_airfoil.ipynb   # ~5 min: retrains TD3 (50k steps, ~2.5 min)
jupyter nbconvert --to notebook --execute --inplace 03_airfrans_cfd_surrogate.ipynb   # ~8 min: ~40 surrogate fits of ~10 s each
jupyter nbconvert --to notebook --execute --inplace 04_airfoil_gust_control_cfd_surrogate.ipynb   # ~5 min: retrains SAC (50k steps, ~3 min)
```

Retraining everything from scratch (delete `models/`) adds about one minute per Stage 1 agent (seven PPO agents of 100,000 steps), two to three minutes per Stage 2 and Stage 4 agent (fourteen TD3/SAC agents of 50,000 steps; PPO takes 10 s), as recorded in the `monitor.csv` logs. The Stage 1 animation cells need the snapshot pickle written during training, so they only run on a training pass.

## Data

The coefficient table `data/airfrans_4digit_coefficients.csv` was extracted from the AirfRANS dataset (Bonnet et al., *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions*, NeurIPS 2022, https://airfrans.readthedocs.io/): one row per four-digit NACA simulation with its Reynolds number, angle of attack, the three NACA parameters, the lift and drag coefficients and a train/test split column (489 rows, 378 train and 111 test). Part A of the Stage 3 notebook, which rebuilds this table from the full dataset (about 9 GB), is switched off; the notebook reads the table instead. The surrogate files in `data/` are the coefficient surrogate that Stage 4 and the companion repository rely on.

## Limitations

- **One seed per configuration.** Every agent and every experiment row is a single training run; the ± in the tables is the spread over evaluation states, not over seeds. Stage 3 is the exception (three seeds per configuration in its experiments).
- **Quasi-steady aerodynamics.** The surrogate maps a steady angle of attack to steady coefficients; the unsteady response is a first-order lag (τ = 0.18 s) and the flap enters as an equivalent angle of attack (k_f = 0.55). There is no Theodorsen or Wagner physics, no stall dynamics and no flap-induced camber in the CFD data.
- **The surrogate is only trusted inside its data.** Training angles span −4.9° to 14.8° and Reynolds numbers 2–6 million; outside that band the extrapolation experiments of Stage 3 show the error growing between about two- and six-fold, and the domain-protection experiment of Stage 4 shows what happens when a controller drives the model there.
- **Small models and budgets.** 50,000–100,000 environment steps, default Stable-Baselines3 networks, no hyperparameter search; the algorithm comparisons are indicative, not conclusive (the main Stage 4 agent, SAC, is beaten by PPO and TD3 on the suite).
- **The gust is a prescribed one-minus-cosine profile** observed exactly by the controller (its angle is in the observation); the "no gust observation" experiment of Stage 2 shows the effect of removing it.
- **Stage 3's five-digit experiment was not run** (it needs the full 9 GB dataset); `results/03_airfrans_cfd_surrogate/ex6_five_digit_NOT_RUN.txt` records this.
- **Exact digits are platform-dependent for Stage 1.** The chaotic attractor amplifies FFT round-off, so the Stage 1 costs reproduce to three decimals on the original machine and only in their order of magnitude elsewhere; the tests assert accordingly.
