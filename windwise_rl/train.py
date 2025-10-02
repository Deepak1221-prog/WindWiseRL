import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.wind_farm_env import WindFarmEnv

def train_model(total_timesteps=20000, num_turbines=10, grid_size=20):
    """
    Train PPO model for wind farm optimization.
    Returns the trained model path.
    """
    env = DummyVecEnv([lambda: WindFarmEnv(grid_size=grid_size, num_turbines=num_turbines)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models", exist_ok=True)
    model_path = "models/ppo_windfarm.zip"
    model.save(model_path)
    print(f"✅ Saved model to {model_path}")
    return model_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=20000)
    parser.add_argument("--num-turbines", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=20)
    args = parser.parse_args()

    train_model(args.total_timesteps, args.num_turbines, args.grid_size)

if __name__ == "__main__":
    main()
