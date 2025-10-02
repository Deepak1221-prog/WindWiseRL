import streamlit as st
import pandas as pd
import os
from train import train_model
from evaluate import evaluate_model
from baseline import grid_baseline
from sim.wake_model import power_from_wind_speed, apply_wake_losses
from utils.data_loader import compute_mean_wind

st.set_page_config(page_title="Wind Farm Optimizer", layout="wide")
st.title("🌬️ AI-Optimized Wind Farm Layout Designer")

# Sidebar controls
st.sidebar.header("Simulation Settings")
grid_size = st.sidebar.slider("Grid Size", 10, 50, 20)
num_turbines = st.sidebar.slider("Number of Turbines", 5, 50, 10)
timesteps = st.sidebar.number_input("Training Timesteps", value=20000, step=1000)

# File upload
st.sidebar.header("Upload Datasets")
terrain_file = st.sidebar.file_uploader("Upload Terrain CSV", type="csv")
wind_file = st.sidebar.file_uploader("Upload Wind CSV", type="csv")

# Save uploaded files to data folder
if terrain_file is not None:
    with open("data/terrain.csv", "wb") as f:
        f.write(terrain_file.getbuffer())
    st.sidebar.success("✅ Terrain file uploaded")

if wind_file is not None:
    with open("data/wind.csv", "wb") as f:
        f.write(wind_file.getbuffer())
    st.sidebar.success("✅ Wind file uploaded")

# Tabs for workflow
tab1, tab2 = st.tabs(["🚀 Train & Evaluate", "⚖️ Compare with Baseline"])

with tab1:
    if st.button("🚀 Start Training & Evaluation"):
        with st.spinner("Training model... please wait."):
            model_path = train_model(
                total_timesteps=timesteps,
                num_turbines=num_turbines,
                grid_size=grid_size
            )
        st.success(f"Model training complete! ✅ Saved at {model_path}")

        # Auto-run evaluation after training
        with st.spinner("Evaluating optimized layout..."):
            evaluate_model(model_path, grid_size=grid_size, num_turbines=num_turbines)

        # Show results
        if os.path.exists("cad_output/turbine_coords.csv"):
            coords = pd.read_csv("cad_output/turbine_coords.csv")
            st.subheader("Optimized Turbine Coordinates")
            st.dataframe(coords)

            # Total energy metric
            if "annual_energy_MWh" in coords.columns:
                total_energy = coords["annual_energy_MWh"].iloc[0]
                st.metric("⚡ Total Annual Energy (MWh/year)", f"{total_energy:.2f}")

            # Layout visualization
            if os.path.exists("cad_output/layout.png"):
                st.subheader("Optimized Layout Visualization")
                st.image("cad_output/layout.png", use_column_width=True)

            # Download buttons
            st.download_button(
                label="⬇️ Download Layout CSV",
                data=coords.to_csv(index=False),
                file_name="turbine_coords.csv",
                mime="text/csv"
            )
            if os.path.exists("cad_output/layout.png"):
                with open("cad_output/layout.png", "rb") as f:
                    st.download_button(
                        label="⬇️ Download Layout Image",
                        data=f,
                        file_name="layout.png",
                        mime="image/png"
                    )

with tab2:
    if st.button("⚖️ Compare RL vs. Baseline"):
        # Generate baseline layout
        grid_baseline(grid_size=grid_size, num_turbines=num_turbines)
        baseline_path = "cad_output/baseline_grid.csv"
        rl_path = "cad_output/turbine_coords.csv"

        if os.path.exists(baseline_path) and os.path.exists(rl_path):
            base_coords = pd.read_csv(baseline_path)
            rl_coords = pd.read_csv(rl_path)

            # Compute baseline annual energy
            mean_wind_map, _ = compute_mean_wind("data/wind.csv")
            eff_map_base = apply_wake_losses(list(zip(base_coords.x, base_coords.y)), mean_wind_map, 270.0)
            eff_map_rl = apply_wake_losses(list(zip(rl_coords.x, rl_coords.y)), mean_wind_map, 270.0)

            hours_per_year = 8760
            baseline_energy = sum(
                power_from_wind_speed(eff_map_base.get((x, y), 0.0)) * hours_per_year / 1000.0
                for x, y in zip(base_coords.x, base_coords.y)
            )
            rl_energy = sum(
                power_from_wind_speed(eff_map_rl.get((x, y), 0.0)) * hours_per_year / 1000.0
                for x, y in zip(rl_coords.x, rl_coords.y)
            )

            improvement = ((rl_energy - baseline_energy) / baseline_energy) * 100

            st.metric("Baseline Annual Energy (MWh/year)", f"{baseline_energy:.2f}")
            st.metric("Optimized Annual Energy (MWh/year)", f"{rl_energy:.2f}")
            st.metric("🚀 Improvement (%)", f"{improvement:.2f}%")
