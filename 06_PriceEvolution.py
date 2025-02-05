# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 09:14:36 2022

@author: jones

"""

#%% Function setup

# Import libraries
import os
import sys
import pypsa
import numpy as np
import pandas as pd
import time
import math

# Timer
t0 = time.time() # Start a timer

# Import functions file
sys.path.append(os.path.split(os.getcwd())[0])
from functions_file import *    

#%% Setup paths

# Directory of files
directory = os.path.split(os.getcwd())[0] + "\\Data\\elec_only\\"

# Figure path
figurePath = os.getcwd() + "\\Figures\\"

# List of file names (CO2)
filename_CO2 = [
                #"postnetwork-elec_only_0.125_0.6.h5",
                #"postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"
                ]

# List of file names (Transmission)
filename_trans = ["postnetwork-elec_only_0_0.05.h5",
                  "postnetwork-elec_only_0.0625_0.05.h5",
                  "postnetwork-elec_only_0.125_0.05.h5",
                  "postnetwork-elec_only_0.25_0.05.h5",
                  "postnetwork-elec_only_0.375_0.05.h5"]


#%% Setup constrain names

# List of constraints
#constraints_CO2 = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]
constraints_CO2 = ["60%", "70%", "80%", "90%", "95%"]
constraints_trans = ["Zero", "Current", "2x Current", "4x Current", "6x Current"]

#%% Quantiles data collection - CO2 constraint

# Variable to store nodal price mean and standard variation
meanPriceElec = []
stdMeanPriceElec = []
meanPriceElecWC = []
meanPriceElecWT = []
quantileMeanPriceElecC = []
quantileMinPriceElecC = []
quantileMeanPriceElecT = []
quantileMinPriceElecT = []
quantileMeanPriceElecWC = []
quantileMeanPriceElecWT = []

# For loop
for file in filename_CO2:
    # Network
    network = pypsa.Network(directory + file)

    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Country weight
    sizeWeight = network.loads_t.p_set.sum() / network.loads_t.p_set.sum().sum()
    
    # Prices for electricity for each country (restricted to 1000 €/MWh)
    priceElec = FilterPrice(network.buses_t.marginal_price[dataNames], 465)
    

    # ----------------------- NP Mean (Elec) --------------------#
    # --- Elec ---
    # Mean price for country
    minPrice = priceElec.min().mean()
    meanPrice = priceElec.mean().mean()
    
    # weighted average
    meanPriceWC = np.average(np.average(priceElec,axis=0), weights=sizeWeight)
    meanPriceWT = np.average(np.average(priceElec,axis=1,weights=sizeWeight))
    
    # append min, max and mean to matrix
    meanPriceElec.append([minPrice, meanPrice])
    meanPriceElecWC.append([minPrice, meanPriceWC])
    meanPriceElecWT.append([minPrice, meanPriceWT])
    
    
    # ----------------------- NP Quantile (Elec) --------------------#
    # --- Elec ---
    # Mean price for flat country
    quantileMinPriceC = np.quantile(priceElec.min(),[0.05,0.25,0.75,0.95])
    quantileMeanPriceC = np.quantile(priceElec.mean(),[0.05,0.25,0.75,0.95])
    
    # Mean price for flat time
    quantileMinPriceT = np.quantile(priceElec.min(axis=1),[0.05,0.25,0.75,0.95])
    quantileMeanPriceT = np.quantile(priceElec.mean(axis=1),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight country
    quantileMeanPriceWC = np.quantile(np.average(priceElec,axis=0),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight time
    quantileMeanPriceWT = np.quantile(np.average(priceElec,axis=1,weights=sizeWeight),[0.05,0.25,0.75,0.95]) 
    
    # append min and mean to matrix (flat country)
    quantileMeanPriceElecC.append(quantileMeanPriceC)
    quantileMinPriceElecC.append(quantileMinPriceC)
    
    # append min and mean to matrix (flat time)
    quantileMeanPriceElecT.append(quantileMeanPriceT)
    quantileMinPriceElecT.append(quantileMinPriceT)
    
    # append min and mean to matrix (weight country)
    quantileMeanPriceElecWC.append(quantileMeanPriceWC)
    
    # append min and mean to matrix (weight time)
    quantileMeanPriceElecWT.append(quantileMeanPriceWT)

#%% Plot quantiles - CO2 constraint

# ----------------------- Price evalution (Elec) (flat country) --------------------#
title =  "Elec NP CO2 Evolution (flat average across countries)"
fig = PriceEvolution(meanPriceElec,quantileMeanPriceElecC, quantileMinPriceElecC, ylim=185, constraintype=constraints_CO2, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  (file[12:-14] + " - Elec NP CO2 Evolution")
path = figurePath + "flat average across countries\\"
SavePlot(fig, path, title)

# ----------------------- Price evalution (Elec) (flat time) --------------------#
title =  "Elec NP CO2 Evolution (flat average across time)"
fig = PriceEvolution(meanPriceElec,quantileMeanPriceElecT, quantileMinPriceElecT, ylim=185, constraintype=constraints_CO2, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  (file[12:-14] + " - Elec NP CO2 Evolution")
path = figurePath + "flat average across time\\"
SavePlot(fig, path, title)

# ----------------------- Price evalution (Elec) (weighted country) --------------------#
title =  "Elec NP CO2 Evolution (weighted average across countries)"
fig = PriceEvolution(meanPriceElecWC,quantileMeanPriceElecWC, quantileMinPriceElecC, ylim=185, constraintype=constraints_CO2, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  (file[12:-14] + " - Elec NP CO2 Evolution")
path = figurePath + "weighted average across countries\\"
SavePlot(fig, path, title)

# ----------------------- Price evalution (Elec) (weighted time) --------------------#
title =  "Elec NP CO2 Evolution (weighted average across time)"
fig = PriceEvolution(meanPriceElecWT,quantileMeanPriceElecWT, quantileMinPriceElecT, ylim=185, constraintype=constraints_CO2, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  (file[12:-14] + " - Elec NP CO2 Evolution")
path = figurePath + "weighted average across time\\"
SavePlot(fig, path, title)

#%% Quantiles data collection - Transmission constraint

# Variable to store nodal price mean and standard variation
meanPriceElec = []
stdMeanPriceElec = []
meanPriceElecWC = []
meanPriceElecWT = []
quantileMeanPriceElecC = []
quantileMinPriceElecC = []
quantileMeanPriceElecT = []
quantileMinPriceElecT = []
quantileMeanPriceElecWC = []
quantileMeanPriceElecWT = []

# For loop
for file in filename_trans:
    # Network
    network = pypsa.Network(directory + file)

    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Country weight
    sizeWeight = network.loads_t.p_set.sum() / network.loads_t.p_set.sum().sum()
    
    # Prices for electricity for each country (restricted to 1000 €/MWh)
    priceElec = FilterPrice(network.buses_t.marginal_price[dataNames], 465)
    

    # ----------------------- NP Mean (Elec) --------------------#
    # --- Elec ---
    # Mean price for country
    minPrice = priceElec.min().mean()
    meanPrice = priceElec.mean().mean()
    
    # weighted average
    meanPriceWC = np.average(np.average(priceElec,axis=0), weights=sizeWeight)
    meanPriceWT = np.average(np.average(priceElec,axis=1,weights=sizeWeight))
    
    # append min, max and mean to matrix
    meanPriceElec.append([minPrice, meanPrice])
    meanPriceElecWC.append([minPrice, meanPriceWC])
    meanPriceElecWT.append([minPrice, meanPriceWT])
    
    
    # ----------------------- NP Quantile (Elec) --------------------#
    # --- Elec ---
    # Mean price for flat country
    quantileMinPriceC = np.quantile(priceElec.min(),[0.05,0.25,0.75,0.95])
    quantileMeanPriceC = np.quantile(priceElec.mean(),[0.05,0.25,0.75,0.95])
    
    # Mean price for flat time
    quantileMinPriceT = np.quantile(priceElec.min(axis=1),[0.05,0.25,0.75,0.95])
    quantileMeanPriceT = np.quantile(priceElec.mean(axis=1),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight country
    quantileMeanPriceWC = np.quantile(np.average(priceElec,axis=0),[0.05,0.25,0.75,0.95])  
    
    # Mean price for weight time
    quantileMeanPriceWT = np.quantile(np.average(priceElec,axis=1,weights=sizeWeight),[0.05,0.25,0.75,0.95]) 
    
    # append min and mean to matrix (flat country)
    quantileMeanPriceElecC.append(quantileMeanPriceC)
    quantileMinPriceElecC.append(quantileMinPriceC)
    
    # append min and mean to matrix (flat time)
    quantileMeanPriceElecT.append(quantileMeanPriceT)
    quantileMinPriceElecT.append(quantileMinPriceT)
    
    # append min and mean to matrix (weight country)
    quantileMeanPriceElecWC.append(quantileMeanPriceWC)
    
    # append min and mean to matrix (weight time)
    quantileMeanPriceElecWT.append(quantileMeanPriceWT)

#%% Plot quantiles - Transmission constraint

# ----------------------- Price evalution (Elec) --------------------#
title = "Elec NP Trans Evolution (flat average across countries)"
fig = PriceEvolution(meanPriceElec,quantileMeanPriceElecC, quantileMinPriceElecC, ylim=185, networktype="green", title=title, figsize=[6,3.2], fontsize=16)
title =  (file[12:-14] + " - Elec NP Trans Evolution")
path = figurePath + "flat average across countries\\"
SavePlot(fig, path, title)

# ----------------------- Price evalution (Elec) --------------------#
title = "Elec NP Trans Evolution (flat average across time)"
fig = PriceEvolution(meanPriceElec,quantileMeanPriceElecT, quantileMinPriceElecT, ylim=185, networktype="green", title=title, figsize=[6,3.2], fontsize=16)
title =  (file[12:-14] + " - Elec NP Trans Evolution")
path = figurePath + "flat average across time\\"
SavePlot(fig, path, title)

# ----------------------- Price evalution (Elec) (weighted country) --------------------#
title =  "Elec NP Trans Evolution (weighted average across countries)"
fig = PriceEvolution(meanPriceElecWC,quantileMeanPriceElecWC, quantileMinPriceElecC, ylim=185, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  (file[12:-14] + " - Elec NP Trans Evolution")
path = figurePath + "weighted average across countries\\"
SavePlot(fig, path, title)

# ----------------------- Price evalution (Elec) (weighted time) --------------------#
title =  "Elec NP Trans Evolution (weighted average across time)"
fig = PriceEvolution(meanPriceElecWT,quantileMeanPriceElecWT, quantileMinPriceElecT, ylim=185, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  (file[12:-14] + " - Elec NP Trans Evolution")
path = figurePath + "weighted average across time\\"
SavePlot(fig, path, title)