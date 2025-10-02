import argparse
import os
import numpy as np
import pandas as pd
from env.wind_farm_env import WindFarmEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from sim.wake_model import power_from_wind_speed, apply_wake_losses
from utils.data_loader import compute_mean_wind


def evaluate_model(model_path, grid_size=20, num_turbines=10, visualize=True):
    # Wrap env for SB3
    env = DummyVecEnv([lambda: WindFarmEnv(grid_size=grid_size, num_turbines=num_turbines)])
    model = PPO.load(model_path, env=env)

    obs = env.reset()
    done = False
    placements = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        a = int(action[0])
        x = a // grid_size
        y = a % grid_size
        placements.append((x, y))
        done = bool(dones[0])

    # Remove duplicates, trim to num_turbines
    unique = []
    for p in placements:
        if p not in unique:
            unique.append(p)
    unique = unique[:num_turbines]

    # Save to CSV
    os.makedirs("cad_output", exist_ok=True)
    df = pd.DataFrame(unique, columns=["x", "y"])

    # Add elevation
    try:
        terrain = pd.read_csv("data/terrain.csv")
        elevs = []
        for x, y in unique:
            elev = terrain[(terrain.x == x) & (terrain.y == y)].elevation.values
            elevs.append(float(elev[0]) if len(elev) > 0 else 0.0)
        df["elevation"] = elevs
    except Exception:
        df["elevation"] = 0.0

    # Compute annual energy using mean wind + wake model
    mean_wind_map, _ = compute_mean_wind("data/wind.csv")
    eff_map = apply_wake_losses(unique, mean_wind_map, 270.0)
    hours_per_year = 8760
    total_energy = 0.0
    for pos in unique:
        v = eff_map.get(pos, 0.0)
        p_kw = power_from_wind_speed(v)
        total_energy += p_kw * hours_per_year / 1000.0  # MWh/year

    df["annual_energy_MWh"] = round(total_energy, 2)
    df.to_csv("cad_output/turbine_coords.csv", index=False)

    print("✅ Saved CAD output to cad_output/turbine_coords.csv")
    print("✅ Saved visualization to cad_output/layout.png")
    print(f"⚡ Total Annual Energy from optimized layout: {total_energy:.2f} MWh/year")

    # Visualization
    if visualize:
        try:
            terrain = pd.read_csv("data/terrain.csv")
            max_x = int(terrain.x.max()) + 1
            max_y = int(terrain.y.max()) + 1
            grid = np.zeros((max_x, max_y))
            for _, r in terrain.iterrows():
                grid[int(r.x), int(r.y)] = r.elevation

            plt.figure(figsize=(6, 6))
            plt.imshow(grid.T, origin="lower")
            xs = [p[0] for p in unique]
            ys = [p[1] for p in unique]
            plt.scatter(xs, ys, marker="x", c="red")
            plt.title("Optimized Turbine Layout")
            plt.savefig("cad_output/layout.png")
        except Exception as e:
            print("Visualization failed:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/ppo_windfarm.zip")
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--num-turbines", type=int, default=10)
    args = parser.parse_args()
    evaluate_model(args.model_path, grid_size=args.grid_size, num_turbines=args.num_turbines)
