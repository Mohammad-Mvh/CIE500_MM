# %%
from pyswmm import Simulation, Subcatchments, Nodes, Links
import numpy as np
import matplotlib.pyplot as plt
import swmmio
import pandas as pd
from datetime import datetime, timedelta

# %%
# import SWMM model
model = swmmio.Model("BellingeSWMM_v021_nopervious.inp")
model.summary
# show data
# model.inp._sections
# model.inp.subcatchments
# model.inp.outfalls
# model.inp.raingages
# model.inp.vertices
# model.inp.orifices


# %%
# Modify the model so all subcatchments use the same raingage with 5-min time step
# Update the 'Raingage' field for all subcatchments
model.inp.subcatchments["Raingage"] = "rg5425"
# Remove the second raingage
if "rg5427" in model.inp.raingages.index:
    model.inp.raingages = model.inp.raingages.drop("rg5427")
print(model.inp.raingages)  # Check if the raingage is removed
# change the rainfall time step to 5 minutes
model.inp.raingages.loc["rg5425", "TimeIntrvl"] = "00:05"
# %%
# Input new rainfall data

# Load your original rainfall CSV (minutes from start and rainfall depth)
df = pd.read_csv("rain_5min.csv", header=None, names=["Minute", "Rainfall"])

# Set the correct start time to match the .inp file
start_time = datetime(2012, 6, 29, 0, 1)

# Generate correct datetime stamps
df["Datetime"] = df["Minute"].apply(
    lambda x: start_time + timedelta(minutes=x))
df["Gage"] = "rg5425"
df["Year"] = df["Datetime"].dt.year
df["Month"] = df["Datetime"].dt.month
df["Day"] = df["Datetime"].dt.day
df["Hour"] = df["Datetime"].dt.hour
df["MinuteOnly"] = df["Datetime"].dt.minute

# Final format as expected by SWMM
df_final = df[["Gage", "Year", "Month",
               "Day", "Hour", "MinuteOnly", "Rainfall"]]
df_final.to_csv("rg_bellinge_Jun2010_Aug2021.dat",
                sep='\t', index=False, header=False)

print(model.inp.raingages.loc["rg5425"])
# %%
# Save the new model
model.inp.save('BellingeSWMM_5min.inp')
# %%
