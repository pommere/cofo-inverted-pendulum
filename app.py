import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import io

# 1. Page Branding & UI
st.set_page_config(page_title="Inverted Pendulum Lab", layout="centered")
st.title("Inverted Pendulum Lab")
st.markdown("""
Welcome to the Physics Lab! Students deduce the local acceleration due to gravity ($g$)
by modeling human locomotion as an **inverted pendulum**. The validity of this model
is explored by examining the **Froude Number** ($Fr$) constraints and biological noise found
in their own gait.
1. Upload your **Phyphox CSV** file below.
2. The app will calculate the FFT (which converts your walking motion from a time signal into a frequency spectrum) to estimate $g$ from your stride period.
""")

# --- 2. Sidebar: Biometrics & Environment ---
st.sidebar.header("1. Anatomical Measurements")
h_hip = st.sidebar.number_input("Floor to Hip (Greater Trochanter) [cm]", value=90.0, help="Pivot point for the inverted pendulum model.")
h_ankle = st.sidebar.number_input("Floor to Ankle (Talus) [cm]", value=8.0, help="Height of the 'foot' pivot above the floor.")

st.sidebar.header("2. Gravity Settings")
local_g = st.sidebar.number_input("Local Gravity (m/s²)", value=9.806, help="Standard Earth gravity is 9.806. Point Lookout is ~9.80.")
env_choice = st.sidebar.selectbox("Simulate Walking On:", ["Earth", "Moon", "Mars", "Jupiter"])

# Planetary gravity constants for comparison
env_g_map = {"Earth": 9.806, "Moon": 1.62, "Mars": 3.71, "Jupiter": 24.79}
sim_g = env_g_map[env_choice]

# --- 3. Physics & Math Functions ---
def lorentzian(x, a, x0, gamma):
    """Lorentzian function for fitting the resonance peak."""
    return a * (gamma**2 / ((x - x0)**2 + gamma**2))

def calculate_g_physics(step_freq, hip_cm, ankle_cm):
    """Calculates g using the inverted pendulum model of walking."""
    # L_leg is the distance from ankle to hip pivot
    L_meters = (hip_cm - ankle_cm) / 100.0
    # In this model: T (Stride Period) = 2 / step_frequency
    stride_period = 2 / step_freq
    g_calc = (4 * np.pi**2 * L_meters) / (stride_period**2)
    return g_calc, L_meters

# --- 4. File Upload & Processing ---
uploaded_file = st.file_uploader("Upload your Phyphox CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        # Standard Phyphox column headers
        time = df['Time (s)'].values
        accel = df['Absolute acceleration (m/s^2)'].values

        # --- FFT & Normalization ---
        dt = np.mean(np.diff(time))
        accel_detrended = accel - np.mean(accel)
        fft_values = np.fft.rfft(accel_detrended)
        frequencies = np.fft.rfftfreq(len(accel), d=dt)
        magnitude_norm = np.abs(fft_values) / len(accel)
        magnitude_norm /= np.max(magnitude_norm)

        # --- Peak Finding Logic ---
        search_mask = (frequencies >= 1.0) & (frequencies <= 4.0)
        f_search = frequencies[search_mask]
        m_search = magnitude_norm[search_mask]
        
        peaks, _ = find_peaks(m_search, height=0.1, distance=20)
        f_step_guess = f_search[peaks[0]] if len(peaks) > 0 else f_search[np.argmax(m_search)]

        # --- Lorentzian Fitting ---
        upper_limit = max(3.0, 1.5 * f_step_guess)
        mask = (frequencies >= 0.5) & (frequencies <= upper_limit)
        x_fit, y_fit = frequencies[mask], magnitude_norm[mask]
        
        popt, _ = curve_fit(lorentzian, x_fit, y_fit, p0=[1.0, f_step_guess, 0.1])
        a_fit, f0_fit, gamma_fit = popt

        # --- Final Physics Calculations ---
        calc_g, L_eff = calculate_g_physics(f0_fit, h_hip, h_ankle)
        percent_error = abs(calc_g - local_g) / local_g * 1
