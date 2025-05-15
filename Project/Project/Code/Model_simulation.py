# %%
import os
from pyswmm import Simulation, Nodes
import pandas as pd
from datetime import datetime, timedelta
import swmmio
import numpy as np

# %%
# Load the modified model
model_path = "BellingeSWMM_5min.inp"
model = swmmio.Model(model_path)

# Define outfall IDs manually (since you gave them)
outfall_ids = [
    "F74F360", "G60K220", "G60U02F", "G70U01F", "G71U02F",
    "G71U10F", "G72U07R", "G75U03F", "G80U03F"
]

# Load rainfall time series
rainfall_df = pd.read_csv("RainfallTimeSeries.csv", header=0)  # <-- header=0
minutes = rainfall_df.iloc[:, 0].values.astype(int)
rainfall_scenarios = rainfall_df.iloc[:, 1:]  # Each column is a scenario

# Define simulation start
start_time = datetime(2012, 6, 29, 0, 1)

# Prepare storage for each outfall
outfall_data = {outfall: pd.DataFrame(index=np.arange(
    0, minutes[-1]+5, 5)) for outfall in outfall_ids}

# Loop through each scenario
for scenario_idx, scenario_name in enumerate(rainfall_scenarios.columns):
    print(f"Running Scenario {scenario_idx+1}...")

    # Create rainfall .dat file for current scenario
    df_rain = pd.DataFrame({
        "Minute": minutes,
        "Rainfall": rainfall_scenarios[scenario_name].values
    })
    df_rain["Datetime"] = df_rain["Minute"].apply(
        lambda x: start_time + timedelta(minutes=x))
    df_rain["Gage"] = "rg5425"
    df_rain["Year"] = df_rain["Datetime"].dt.year
    df_rain["Month"] = df_rain["Datetime"].dt.month
    df_rain["Day"] = df_rain["Datetime"].dt.day
    df_rain["Hour"] = df_rain["Datetime"].dt.hour
    df_rain["MinuteOnly"] = df_rain["Datetime"].dt.minute

    df_final = df_rain[["Gage", "Year", "Month",
                        "Day", "Hour", "MinuteOnly", "Rainfall"]]
    df_final.to_csv("rg_bellinge_Jun2010_Aug2021.dat",
                    sep="\t", index=False, header=False)

    # Run SWMM simulation
    with Simulation(model_path) as sim:
        nodes = Nodes(sim)

        # Prepare temp storage
        temp_storage = {outfall: [] for outfall in outfall_ids}
        time_storage = []

        for step in sim:
            current_time = sim.current_time
            elapsed_minutes = int(
                (current_time - start_time).total_seconds() / 60)

            time_storage.append(elapsed_minutes)
            for outfall in outfall_ids:
                flow = nodes[outfall].total_inflow
                temp_storage[outfall].append(flow)

    # Organize and interpolate outfall discharges
    sim_times = np.array(time_storage)
    for outfall in outfall_ids:
        series = pd.Series(temp_storage[outfall], index=sim_times)
        series = series[~series.index.duplicated(
            keep="first")]  # Remove duplicates

        # Reindex to regular 5-min intervals and interpolate
        series_regular = series.reindex(np.arange(0, minutes[-1]+5, 5))
        series_regular = series_regular.interpolate(method="linear")

        # Store in the main DataFrame
        outfall_data[outfall][f"Scenario_{scenario_idx+1}"] = series_regular

# %%
# Save to a single Excel file with multiple sheets
output_dir = "outfall_flow_results_csv"
os.makedirs(output_dir, exist_ok=True)

for outfall, df_outfall in outfall_data.items():
    df_outfall.to_csv(os.path.join(output_dir, f"{outfall}.csv"), index=False)

print(f"\n✅ Simulation complete! Results saved as CSVs in '{output_dir}'")

# %%
