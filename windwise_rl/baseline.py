
import pandas as pd, numpy as np, os
def grid_baseline(grid_size=20, num_turbines=10):
    xs = np.linspace(0, grid_size-1, int(np.ceil(np.sqrt(num_turbines)))).astype(int)
    ys = np.linspace(0, grid_size-1, int(np.ceil(np.sqrt(num_turbines)))).astype(int)
    pts = []
    for i in xs:
        for j in ys:
            pts.append((int(i), int(j)))
            if len(pts) >= num_turbines:
                break
        if len(pts) >= num_turbines:
            break
    df = pd.DataFrame(pts, columns=['x','y'])
    os.makedirs('cad_output', exist_ok=True)
    df.to_csv('cad_output/baseline_grid.csv', index=False)
    print('Wrote cad_output/baseline_grid.csv')

if __name__ == '__main__':
    grid_baseline()
