# Airfoil gust control with a CFD surrogate and deep reinforcement learning

This repository holds the first four stages of a project in which I take a flap controller for a two-dimensional airfoil from a textbook aerodynamic law to a neural-network surrogate built on RANS simulations. Stage 1 sets up the method on a chaotic field, the forced Kuramoto–Sivashinsky equation, wrapped as a Gymnasium environment in which PPO, SAC and TD3 agents from Stable-Baselines3 learn to suppress the chaos with a few actuators. Stage 2 applies the same method to gust rejection on a thin-airfoil model, where the lift follows 2πα and a flap is the control. Stage 3 replaces that law with a multilayer-perceptron surrogate trained on the AirfRANS coefficient table, mapping the Reynolds number, the angle of attack and the three NACA four-digit parameters to the lift and drag coefficients. Stage 4 puts the surrogate inside the control environment and trains the flap controller against it. Every notebook ends with the exercises and the extra experiments I ran, and each has a training animation. The second half of the project, reduced-order models of the flow fields and a design search in their latent space, is in the companion repository `airfoil-rom-betavae-design`.

Portfolio page: https://vamsigudipati.github.io/airfoil-cfd-surrogate-drl.html

## Notebooks

| Notebook | Stage | Results | Models | Animations |
| --- | --- | --- | --- | --- |
| `notebooks/01_ks_control_gymnasium.ipynb` | Stage 1 — Control of a chaotic field: the method | `results/01_ks_control_gymnasium/` | `models/01_ks_control_gymnasium/` | `animations/01_*` |
| `notebooks/02_airfoil_gust_control_thin_airfoil.ipynb` | Stage 2 — Gust control with a thin-airfoil model | `results/02_airfoil_gust_control_thin_airfoil/` | `models/02_airfoil_gust_control_thin_airfoil/` | `animations/02_*` |
| `notebooks/03_airfrans_cfd_surrogate.ipynb` | Stage 3 — A CFD surrogate from AirfRANS | `results/03_airfrans_cfd_surrogate/` | `models/03_airfrans_cfd_surrogate/` | `animations/03_*` |
| `notebooks/04_airfoil_gust_control_cfd_surrogate.ipynb` | Stage 4 — The flap controller trained inside the surrogate | `results/04_airfoil_gust_control_cfd_surrogate/` | `models/04_airfoil_gust_control_cfd_surrogate/` | `animations/04_*` |

Each notebook is also exported as HTML next to it, with all outputs, for reading without a kernel. The `results/` folders hold the tables (CSV) and figures (PNG) of the exercises; the `models/` folders hold the trained agents and surrogates (Stable-Baselines3 `.zip` files with their training logs, Keras `.keras` files with their scalers); `animations/` holds the training and flight videos (MP4) and their final frames.

## Setting up and running

I ran everything on a laptop CPU (Apple M3 Max) with Python 3.12. The frozen environment is in `requirements.txt`; the videos need `ffmpeg` on the path.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name airfoil-drl
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_ks_control_gymnasium.ipynb
```

All paths inside the notebooks are relative to `notebooks/`, so run them from there (or open them in Jupyter from that folder). The exercise agents and the Stage 1 agent are reloaded from `models/` when the files are present; the main agents of Stages 2 and 4 and the surrogate of Stage 3 are retrained by their notebooks (a few minutes each). The snapshot files that the animation cells read are written during training and are not in the repository; to rebuild the Stage 1 animation, move `models/01_ks_control_gymnasium/ppo/` aside so that the agent trains again. Stage 3 loads `data/airfrans_cl_cd_surrogate.keras` and `data/airfrans_cl_cd_scalers.joblib` as the reference surrogate in its last section, and Stage 4 uses the same two files as its aerodynamic law.

## Data

The coefficient table `data/airfrans_4digit_coefficients.csv` was extracted from the AirfRANS dataset (Bonnet et al., *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions*, NeurIPS 2022, https://airfrans.readthedocs.io/): one row per four-digit NACA simulation with its Reynolds number, angle of attack, the three NACA parameters, the lift and drag coefficients and a train/test split column. Part A of the Stage 3 notebook, which rebuilds this table from the full dataset (about 9 GB), is switched off; the notebook reads the table instead. The surrogate files in `data/` are the coefficient surrogate that Stage 4 and the companion repository rely on.
