#!/usr/bin/env python
"""Re-score the saved agents and the provided surrogate on CPU, without any training.

Reproduces, in well under a minute, the headline numbers of the README:

    python scripts/evaluate_pretrained.py            # all stages
    python scripts/evaluate_pretrained.py --stage 2  # one stage

Stage 1 regenerates the chaotic attractor (seed 1234) and scores the saved PPO
agent against no control; Stage 2 flies the saved TD3 agent through the
evaluation gusts; Stage 3 scores the provided surrogate on the AirfRANS test
split; Stage 4 flies the saved SAC agent through the surrogate environment.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], action="append",
                        help="stage(s) to score; default: all four")
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="repository root (default: this checkout)")
    return parser


def stage1(root):
    import numpy as np
    from stable_baselines3 import PPO
    from airfoil_drl.ks import KSSystem

    sys22 = KSSystem()
    pool = sys22.attractor()
    training_pool, test_pool = pool[:400], pool[500:]
    scale = max(float(np.std(training_pool)), 0.1)
    agent = PPO.load(root / "models/01_ks_control_gymnasium/ppo/ks_ppo_final.zip", device="cpu")

    def cost(initial, controller, horizon=80, effort_weight=0.01):
        state, total = np.asarray(initial, np.float32).copy(), 0.0
        for _ in range(horizon):
            a = np.clip(controller(state), -1.0, 1.0)
            state = sys22.step(state, a)
            total += np.mean(state**2) + effort_weight * np.sum(a**2)
        return total

    def ppo(state):
        obs = np.clip(np.asarray(state, np.float32) / scale, -10.0, 10.0)
        return np.asarray(agent.predict(obs, deterministic=True)[0], np.float32)

    initials = test_pool[::5]
    print("Stage 1 — cost over 20 unseen chaotic states, 80 steps (notebook: no control 114.33, PPO 14.63)")
    for name, ctrl in (("no control", lambda s: np.zeros(5, np.float32)), ("PPO", ppo)):
        c = [cost(u0, ctrl) for u0 in initials]
        print(f"  {name:11s}: {np.mean(c):8.3f} +/- {np.std(c):.3f}")


def stage2(root):
    from stable_baselines3 import TD3
    from airfoil_drl.metrics import episode_metrics
    from airfoil_drl.thin_airfoil import AirfoilGymEnv, ZeroPolicy, run_episode, suite_metrics

    agent = TD3.load(root / "models/02_airfoil_gust_control_thin_airfoil/airfoil_0d_td3_final.zip", device="cpu")
    print("Stage 2 — fixed gust, lift RMSE (notebook: uncontrolled 0.13852, TD3 0.01663)")
    for name, pol in (("uncontrolled", ZeroPolicy()), ("TD3", agent)):
        m = episode_metrics(run_episode(AirfoilGymEnv(lift_model="linear", random_gust=False), pol, seed=123))
        print(f"  {name:12s}: CL_RMSE {m['CL_RMSE']:.5f}  max|flap| {m['max_abs_flap_deg']:.2f} deg  mean CD {m['mean_CD']:.5f}")
    print(f"  TD3 mean RMSE over the 5-gust suite: {suite_metrics(agent, 'TD3')['mean_RMSE']:.5f} (csv: 0.01654)")


def stage3(root):
    import numpy as np
    import pandas as pd
    from airfoil_drl.surrogate import load_numpy_surrogate, score

    f = load_numpy_surrogate(root / "data/airfrans_cl_cd_surrogate.keras", root / "data/airfrans_cl_cd_scalers.joblib")
    df = pd.read_csv(root / "data/airfrans_4digit_coefficients.csv")
    test = df[df.split == "test"]
    s = score(lambda fr: np.array([f(1e6 * r.Re_M, r.alpha_deg, r.m, r.p, r.t) for r in fr.itertuples()]), test)
    print("Stage 3 — provided surrogate on the 111 AirfRANS test cases (csv: CL RMSE 0.02593, CD RMSE 0.00087)")
    print(f"  CL RMSE {s['CL_RMSE']:.5f}  R2 {s['CL_R2']:.4f} | CD RMSE {s['CD_RMSE']:.5f}  R2 {s['CD_R2']:.4f}")
    cl, cd = f(4.0e6, 5.0, 2.0, 4.0, 12.0)
    print(f"  NACA 2412, Re 4e6, alpha 5 deg: CL {cl:.4f}, CD {cd:.5f} (notebook: 0.7595, 0.00967)")


def stage4(root):
    import numpy as np
    from stable_baselines3 import SAC
    from airfoil_drl.metrics import episode_metrics
    from airfoil_drl.surrogate import load_numpy_surrogate
    from airfoil_drl.surrogate_env import AirfoilSurrogateGymEnv
    from airfoil_drl.thin_airfoil import ZeroPolicy

    f = load_numpy_surrogate(root / "data/airfrans_cl_cd_surrogate.keras", root / "data/airfrans_cl_cd_scalers.joblib")
    agent = SAC.load(root / "models/04_airfoil_gust_control_cfd_surrogate/airfoil_surrogate_sac_final.zip", device="cpu")
    print("Stage 4 — fixed gust inside the surrogate, lift RMSE (notebook: uncontrolled 0.13536, SAC 0.05962)")
    for name, pol in (("uncontrolled", ZeroPolicy()), ("SAC", agent)):
        env = AirfoilSurrogateGymEnv(f, random_gust=False)
        obs, _ = env.reset(seed=123)
        data = {k: [] for k in ("cl", "cl_target", "cd", "flap_deg", "reward")}
        done = False
        while not done:
            action, _ = pol.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action); done = term or trunc
            for k in data:
                data[k].append(r if k == "reward" else info[k])
        m = episode_metrics({k: np.asarray(v) for k, v in data.items()})
        print(f"  {name:12s}: CL_RMSE {m['CL_RMSE']:.5f}  max|flap| {m['max_abs_flap_deg']:.2f} deg  mean CD {m['mean_CD']:.5f}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    stages = sorted(set(args.stage or [1, 2, 3, 4]))
    for s in stages:
        {1: stage1, 2: stage2, 3: stage3, 4: stage4}[s](args.repo_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
