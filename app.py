import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# 1. Page Branding & UI
st.set_page_config(page_title="🌍 Inverted Pendulum Lab", layout="centered")
st.title("🌍 Inverted Pendulum Lab")
st.markdown(r"""
Welcome to the Physics Lab! Students deduce the local acceleration due to gravity ($g$)
by modeling human locomotion as an **inverted pendulum**. 
""")

# --- 2. Sidebar: Biometrics & Environment ---
st.sidebar.header("1. Anatomical Measurements")
h_hip = st.sidebar.number_input("Floor to Hip (Greater Trochanter) [cm]", value=90.0)
h_ankle = st.sidebar.number_input("Floor to Ankle (Talus) [cm]", value=8.0)

# Reference gravity for the 'g' calculation
G_EARTH = 9.80665

# --- 3. Physics & Math Functions ---
def lorentzian(x, a, x0, gamma):
    return a * (gamma**2 / ((x - x0)**2 + gamma**2))

def calculate_g_physics(step_freq, hip_cm, ankle_cm):
    L_total = (hip_cm - ankle_cm) / 100.0 
    L_eff = 0.55 * L_total 
    # Stride period is time for two steps
    stride_period = 2 / step_freq
    g_calc = (4 * np.pi**2 * L_eff) / (stride_period**2)
    return g_calc

# --- 4. File Upload & Processing ---
uploaded_file = st.file_uploader("Upload your Phyphox CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Synthetic data for demo
    t = np.linspace(0, 10, 1000)
    synth_accel = 9.8 + 2.0 * np.sin(2 * np.pi * 1.475 * t) + np.random.normal(0, 0.3, 1000)
    df = pd.DataFrame({'Time (s)': t, 'Absolute acceleration (m/s^2)': synth_accel})
    st.info("💡 **Displaying synthetic data.** Upload a CSV to analyze your own walk.")

# --- 5. Data Analysis ---
try:
    time = df.iloc[:, 0].values
    accel = df.iloc[:, 1].values

    dt = np.mean(np.diff(time))
    accel_detrended = accel - np.mean(accel)
    fft_values = np.fft.rfft(accel_detrended)
    frequencies = np.fft.rfftfreq(len(accel), d=dt)
    magnitude_norm = np.abs(fft_values) / len(accel)
    magnitude_norm /= np.max(magnitude_norm)

    # Peak Finding & Fitting
    search_mask = (frequencies >= 1.0) & (frequencies <= 4.0)
    f_search = frequencies[search_mask]
    m_search = magnitude_norm[search_mask]
    peaks, _ = find_peaks(m_search, height=0.1)
    f_step_guess = f_search[peaks[0]] if len(peaks) > 0 else 1.5

    popt, _ = curve_fit(lorentzian, frequencies[search_mask], magnitude_norm[search_mask], p0=[1.0, f_step_guess, 0.1])
    f0_fit = popt[1]

    # Physics Results
    calc_g = calculate_g_physics(f0_fit, h_hip, h_ankle)
    g_ratio = calc_g / G_EARTH

    # --- 6. UI: Results Display ---
    st.subheader("Lab Analysis Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Step Frequency", f"{f0_fit:.2f} Hz")
    c2.metric("Calculated Gravity", f"{calc_g:.2f} m/s²")
    c3.metric("Result in terms of g", f"{g_ratio:.2f} g")

    if 0.85 <= g_ratio <= 1.15:
        st.success(f"Your gait model successfully predicted gravity within 15% of Earth's value.")
    else:
        st.warning(f"Calculated gravity is {g_ratio:.2f}x Earth's gravity. Check your leg length or walking speed.")

    # Visualization
    fig, ax = plt.subplots()
    ax.plot(frequencies, magnitude_norm, label='Data', alpha=0.5)
    ax.plot(frequencies[search_mask], lorentzian(frequencies[search_mask], *popt), 'k--', label='Fit')
    ax.set_xlim(0, 4)
    ax.set_xlabel("Frequency (Hz)")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")
