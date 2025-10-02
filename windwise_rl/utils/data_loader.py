
import pandas as pd

def load_terrain(path):
    df = pd.read_csv(path)
    return df

def compute_mean_wind(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    mean = df.groupby(['x','y']).agg({'wind_speed':'mean', 'wind_dir':'mean'}).reset_index()
    mean_map = {(int(r.x), int(r.y)): float(r.wind_speed) for r in mean.itertuples()}
    mean_dir_map = {(int(r.x), int(r.y)): float(r.wind_dir) for r in mean.itertuples()}
    return mean_map, mean_dir_map
