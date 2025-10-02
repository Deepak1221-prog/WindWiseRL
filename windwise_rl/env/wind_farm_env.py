import gymnasium as gym
import numpy as np
from gymnasium import spaces
from utils.data_loader import load_terrain, compute_mean_wind
from sim.wake_model import power_from_wind_speed, apply_wake_losses


class WindFarmEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, terrain_csv="data/terrain.csv", wind_csv="data/wind.csv",
                 grid_size=20, num_turbines=10, cell_size_m=200.0):
        super().__init__()
        self.grid_size = grid_size
        self.num_turbines = num_turbines
        self.cell_size_m = cell_size_m

        # load terrain & wind
        self.terrain = load_terrain(terrain_csv)
        mean_wind_map, _ = compute_mean_wind(wind_csv)

        # build arrays
        self.n_cells = grid_size * grid_size
        self.elev = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.base_wind = np.zeros((grid_size, grid_size), dtype=np.float32)

        for _, r in self.terrain.iterrows():
            x = int(r.x)
            y = int(r.y)
            if 0 <= x < grid_size and 0 <= y < grid_size:
                self.elev[x, y] = float(r.elevation)

        for (x, y), v in mean_wind_map.items():
            if 0 <= x < grid_size and 0 <= y < grid_size:
                self.base_wind[x, y] = float(v)

        # normalization factors
        self.max_elev = max(1.0, float(self.elev.max()))
        self.max_wind = max(0.1, float(self.base_wind.max()))

        # observation: elevation_norm + base_wind_norm + occupancy
        obs_len = self.n_cells * 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_cells)

        self._reset_internal()

    def _reset_internal(self):
        self.occupancy = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.placed = []
        self.steps = 0
        self.done = False
        self.last_energy = 0.0  # MWh/year

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal()
        return self._get_obs(), {}

    def _get_obs(self):
        elev_n = (self.elev / self.max_elev).flatten()
        wind_n = (self.base_wind / self.max_wind).flatten()
        occ = self.occupancy.flatten().astype(np.float32)
        obs = np.concatenate([elev_n, wind_n, occ]).astype(np.float32)
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        x = int(action) // self.grid_size
        y = int(action) % self.grid_size
        reward = 0.0

        if self.occupancy[x, y] == 1:
            # Penalty for trying to place on an occupied cell
            reward = -0.5
        else:
            # Place turbine
            self.occupancy[x, y] = 1
            self.placed.append((x, y))

            # Use a representative mean wind direction (simplified)
            mean_dir = 270.0  # west wind
            eff_map = apply_wake_losses(
                self.placed,
                {(i, j): float(self.base_wind[i, j])
                 for i in range(self.grid_size)
                 for j in range(self.grid_size)},
                mean_dir,
                cell_size_m=self.cell_size_m
            )

            # Compute annual energy production (MWh/year)
            hours_per_year = 8760
            total_energy = 0.0
            for pos in self.placed:
                v = eff_map.get(pos, 0.0)
                p_kw = power_from_wind_speed(v)  # kW
                total_energy += p_kw * hours_per_year / 1000.0  # convert to MWh

            reward = float(total_energy - self.last_energy)
            self.last_energy = float(total_energy)

        self.steps += 1
        if len(self.placed) >= self.num_turbines or self.steps >= self.n_cells:
            self.done = True

        terminated = self.done
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        print(self.occupancy)
