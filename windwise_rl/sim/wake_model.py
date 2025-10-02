
import math

AIR_DENSITY = 1.225  # kg/m^3
ROTOR_RADIUS = 40.0  # m (80 m diameter)
Cp = 0.4  # power coefficient (assumed)
AREA = math.pi * ROTOR_RADIUS**2

def power_from_wind_speed(v):
    return 0.5 * AIR_DENSITY * AREA * Cp * (v**3) / 1000.0

def apply_wake_losses(turbine_positions, base_wind_map, wind_dir_degrees, cell_size_m=200.0):
    theta = math.radians((wind_dir_degrees + 180) % 360)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    eff = {}
    for pos in turbine_positions:
        eff[pos] = base_wind_map.get(pos, 0.0)
    wake_max_m = 2000.0
    wake_strength = 0.35
    for i, p_i in enumerate(turbine_positions):
        xi, yi = p_i
        for j, p_j in enumerate(turbine_positions):
            if i == j:
                continue
            xj, yj = p_j
            dx = ((xj - xi) * cos_t + (yj - yi) * sin_t) * cell_size_m
            if dx <= 0:
                continue
            lateral = abs((- (xj - xi) * sin_t + (yj - yi) * cos_t) * cell_size_m)
            k = 0.05
            r_wake = ROTOR_RADIUS + k * dx
            if lateral > r_wake:
                continue
            if dx > wake_max_m:
                continue
            deficit = wake_strength * (1.0 - (dx / wake_max_m))**1.5
            eff[p_j] = max(0.0, eff[p_j] * (1.0 - deficit))
    return eff
