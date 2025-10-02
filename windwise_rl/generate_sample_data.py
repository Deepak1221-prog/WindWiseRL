
import numpy as np
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

grid_size = 20
x = np.arange(grid_size)
y = np.arange(grid_size)
xx, yy = np.meshgrid(x, y)
elevation = 100 + 20 * np.sin(xx / 3) * np.cos(yy / 3)
terrain_df = pd.DataFrame({
    'x': xx.flatten(),
    'y': yy.flatten(),
    'elevation': elevation.flatten().round(2)
})
terrain_df.to_csv('data/terrain.csv', index=False)
print('Saved data/terrain.csv ({} rows)'.format(len(terrain_df)))

# wind: 48 hours of hourly wind speeds/directions
timestamps = pd.date_range('2020-01-01', periods=48, freq='H')
records = []
for t in timestamps:
    base_speed = 8 + 2*np.sin(t.hour/24*2*np.pi)
    base_dir = 270 + 20*np.sin(t.hour/24*2*np.pi)
    for i in range(grid_size):
        for j in range(grid_size):
            wind_speed = base_speed + 0.1*i - 0.05*j + np.random.normal(0, 0.5)
            wind_dir = base_dir + np.random.normal(0, 5)
            records.append([t, i, j, round(wind_speed, 2), round(wind_dir, 1)])
wind_df = pd.DataFrame(records, columns=['timestamp','x','y','wind_speed','wind_dir'])
wind_df.to_csv('data/wind.csv', index=False)
print('Saved data/wind.csv ({} rows)'.format(len(wind_df)))
