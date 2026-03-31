import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import io

# 1. Page Branding & UI
st.set_page_config(page_title="Inverted Pendulum Lab", layout="centered")
st.title("🏃‍♂️ Inverted Pendulum Gait Lab")
st.markdown("""
Welcome to the Physics Lab! 
1. Upload your **Phyphox CSV** file below.
2. The app will calculate the FFT and estimate **g**.
""")

# 2. Physics Logic
def lorentzian(x, a, x0, gamma):
    return a * (gamma**2 / ((x - x0)**2 + gamma**2))

def calculate_g(step_freq, height_ft):
    height_m = height_ft * 0.3048
    L_eff = height_m * 0.2385 
    T = 2 / step_freq
    return (4 * np.pi**2 * L_eff) / (T**2)

# 3. Sidebar for Student Inputs
st.sidebar.header("Lab Parameters")
h_min = st.sidebar.number_input("Min Student Height (ft)", value=4.5, step=0.1)
h_max = st.sidebar.number_input("Max Student Height (ft)", value=7.0, step=0.1)

# 4. File Uploader
uploaded_file = st.file_uploader("Drop your Phyphox CSV here", type=["csv"])

if uploaded_file:
    try:
        # Read data
        df = pd.read_csv(uploaded_file)
        # Ensure column names match Phyphox default export
        time = df['Time (s)'].values
        accel = df['Absolute acceleration (m/s^2)'].values

        # FFT & Normalization
        dt = np.mean(np.diff(time))
        accel_detrended = accel - np.mean(accel)
        fft_vals = np.fft.rfft(accel_detrended)
        freqs = np.fft.rfftfreq(len(accel), d=dt)
        mag = np.abs(fft_vals) / len(accel)
        mag /= np.max(mag)

        # Robust Peak Logic
        search_mask = (freqs >= 1.0) & (freqs <= 5.0)
        peaks, _ = find_peaks(mag[search_mask], height=0.1, distance=20)
        f_guess = freqs[search_mask][peaks[0]] if len(peaks) > 0 else freqs[search_mask][np.argmax(mag[search_mask])]

        # Lorentzian Fit
        upper_limit = max(3.0, 1.5 * f_guess)
        mask = (freqs >= 0.5) & (freqs <= upper_limit)
        popt, _ = curve_fit(lorentzian, freqs[mask], mag[mask], p0=[1.0, f_guess, 0.1])
        
        # Calculate g-range
        g_low = calculate_g(popt[1], h_min)
        g_high = calculate_g(popt[1], h_max)

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freqs, mag, color='red', alpha=0.4, label='Raw FFT Data')
        
        x_curve = np.linspace(0.5, upper_limit, 1000)
        ax.plot(x_curve, lorentzian(x_curve, *popt), color='black', lw=2, label=f'Fit: {popt[1]:.3f} Hz')
        
        ax.set_xlim(0, upper_limit)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized Magnitude")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Output to App
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        col1.metric("Step Frequency", f"{popt[1]:.3f} Hz")
        col2.metric("Est. g Range", f"{g_low:.2f} - {g_high:.2f}")
        
        st.info(f"The calculated gravity for a person between {h_min}' and {h_max}' is approximately **{g_low:.2f} to {g_high:.2f} m/s²**.")

    except Exception as e:
        st.error(f"Error processing file: {e}. Check if the CSV format is correct!")
