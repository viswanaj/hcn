# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:14:42 2023

@author: jayashri with help from chatGPT
"""

#importing packages
import matplotlib.pyplot as plt 
import numpy as np 
from lmfit import Model, Parameters
import pyabf
import os

#GUI to select a file.
import  tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
print(file_path)

#plots all sweeps in the chosen ABF filepath
def plot_all_sweeps(abf_file_path):
    abf = pyabf.ABF(abf_file_path)  # Load the ABF file
    
    plt.figure(figsize=(10, 6))  # Set the figure size
    
    for sweep_number in abf.sweepList:  # Iterate through all sweeps
        abf.setSweep(sweep_number, channel=1)  # Set the current sweep
        
        # Plot the sweep data
        plt.plot(abf.sweepX, abf.sweepY, label=f'Sweep {sweep_number}')
    
    plt.xlabel(abf.sweepLabelX)  # Set X-axis label
    plt.ylabel(abf.sweepLabelY)  # Set Y-axis label
    plt.title(f'All Sweeps in {abf_file_path}')  # Set the plot title
    plt.legend()  # Show legend with sweep numbers
    plt.grid(True)  # Add gridlines
    plt.show()  # Show the plot
    
    return abf

# Example usage
abf_file_path = file_path
plot_all_sweeps(abf_file_path)



#single exponential decay for Ih

def fit_and_plot_double_exp_decay(abf_file_path, sweep_number, start_index, end_index):
    abf = pyabf.ABF(abf_file_path)  # Load the ABF file
    
    abf.setSweep(sweep_number, channel = 1)  # Set the current sweep
    
    # Extract the section of data to fit
    x_data = abf.sweepX[start_index:end_index + 1]
    y_data = abf.sweepY[start_index:end_index + 1]
    
    # Define the double exponential decay model
    def double_exp_decay(x, amp1, tau1, amp2, tau2, offset):
        return amp1 * np.exp(-x / tau1) + amp2 * np.exp(-x / tau2) + offset
    
    model = Model(double_exp_decay)
    
    # Create initial parameter guesses
    params = Parameters()
    params.add('amp1', value=max(y_data), min=0)
    params.add('tau1', value=1.0, min=0)
    params.add('amp2', value=min(y_data), max=0)
    params.add('tau2', value=1.0, min=0)
    params.add('offset', value=np.mean(y_data))
    
    # Fit the model to the section of data
    result = model.fit(y_data, x=x_data, params=params)
    
    # Extract the fit parameters
    amp1 = result.params['amp1'].value
    tau1 = result.params['tau1'].value
    amp2 = result.params['amp2'].value
    tau2 = result.params['tau2'].value
    offset = result.params['offset'].value
    
    # Plot the original data and the fitted model for the specified section
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, label='Original Data', linestyle='-', marker='o')
    plt.plot(x_data, result.best_fit, label='Double Exp Decay Fit', linestyle='--')
    
    plt.xlabel(abf.sweepLabelX)
    plt.ylabel(abf.sweepLabelY)
    plt.title(f'Double Exponential Decay Fit (Sweep {sweep_number}, Section {start_index}-{end_index})')
    plt.legend()
    plt.grid(True)
    plt.show()

    return amp1, tau1, amp2, tau2, offset

# Example usage
abf_file_path = file_path
sweep_number = 7  # Replace with the sweep number you want to fit and plot
start_index = 58000  # Replace with the start index of the section you want to fit
end_index = 257600  # Replace with the end index of the section you want to fit

amp1, tau1, amp2, tau2, offset = fit_and_plot_double_exp_decay(abf_file_path, sweep_number, start_index, end_index)

print(f"Amp1: {amp1}")
print(f"Tau1: {tau1}")
print(f"Amp2: {amp2}")
print(f"Tau2: {tau2}")
print(f"Offset: {offset}")

