import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import io

# 1. Page Branding & UI
st.set_page_config(page_title="🌍 Inverted Pendulum Lab", layout="centered")
st.title("🌍 Inverted Pendulum Lab")
st.markdown(r"""
Welcome to the Physics Lab! Students deduce the local acceleration due to gravity ($g$)
by modeling human locomotion as an **inverted pendulum**. The validity of this model
is explored by examining the **Froude Number** ($Fr$) constraints and biological noise found
in their own gait.
1. Upload your **Phyphox CSV** file below.
2. The app will calculate the FFT (which converts your walking motion from a time signal into a frequency spectrum) to estimate $g$ from your stride period.
""")

# --- 2. Sidebar: Biometrics & Environment ---
st.sidebar.header("1. Anatomical Measurements")
h_hip = st.sidebar.number_input("Floor to Hip (Greater Trochanter) [cm]", value=90.0)
h_ankle = st.sidebar.number_input("Floor to Ankle (Talus) [cm]", value=8.0)

st.sidebar.header("2. Gravity Settings")
local_g = st.sidebar.number_input("Local Gravity (m/s²)", value=9.806)

# --- 3. Physics & Math Functions ---
def lorentzian(x, a, x0, gamma):
    return a * (gamma**2 / ((x - x0)**2 + gamma**2))

def calculate_g_physics(step_freq, hip_cm, ankle_cm):
    L_total = (hip_cm - ankle_cm) / 100.0 
    # Matches your lab manual's center of mass (0.55L)
    L_eff = 0.55 * L_total 
    stride_period = 2 / step_freq
    g_calc = (4 * np.pi**2 * L_eff) / (stride_period**2)
    return g_calc, L_total

# --- 4. File Upload & Processing ---
uploaded_file = st.file_uploader("Upload your Phyphox CSV file", type=["csv"])

# Data Source Logic: Use uploaded file if exists, otherwise generate synthetic data
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Successfully loaded your gait data.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    # Generate Synthetic "Ideal" Data (1.25 Hz walking frequency)
    t = np.linspace(0, 10, 1000)
    # 1.25Hz signal simulates a standard brisk walk
    synth_accel = 9.8 + 2.0 * np.sin(2 * np.pi * 1.475 * t) + np.random.normal(0, 0.3, 1000)
    df = pd.DataFrame({'Time (s)': t, 'Absolute acceleration (m/s^2)': synth_accel})
    st.info("💡 **No file uploaded yet.** Displaying synthetic 'Perfect Walk' data (1.475 Hz) as an example.")

# --- 5. Data Analysis & Physics ---
try:
    time = df['Time (s)'].values
    accel = df['Absolute acceleration (m/s^2)'].values

    # FFT & Normalization
    dt = np.mean(np.diff(time))
    accel_detrended = accel - np.mean(accel)
    fft_values = np.fft.rfft(accel_detrended)
    frequencies = np.fft.rfftfreq(len(accel), d=dt)
    magnitude_norm = np.abs(fft_values) / len(accel)
    magnitude_norm /= (np.max(magnitude_norm) if np.max(magnitude_norm) != 0 else 1)

    # Peak Finding Logic
    search_mask = (frequencies >= 1.0) & (frequencies <= 4.0)
    f_search = frequencies[search_mask]
    m_search = magnitude_norm[search_mask]
    peaks, _ = find_peaks(m_search, height=0.1, distance=20)
    f_step_guess = f_search[peaks[0]] if len(peaks) > 0 else f_search[np.argmax(m_search)]

    # Lorentzian Fitting
    upper_limit = max(3.0, 1.5 * f_step_guess)
    mask = (frequencies >= 0.5) & (frequencies <= upper_limit)
    popt, _ = curve_fit(lorentzian, frequencies[mask], magnitude_norm[mask], p0=[1.0, f_step_guess, 0.1])
    f0_fit = popt[1]

    # Physics Results
    calc_g, L_total = calculate_g_physics(f0_fit, h_hip, h_ankle)
    percent_error = abs(calc_g - local_g) / local_g * 100

    # --- 6. UI: Results Display ---
    st.subheader("Lab Analysis Results")
    
    # Calculate gravity as a ratio of the local reference (G-units)
    g_ratio = calc_g / local_g
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Step Frequency ($f_0$)", f"{f0_fit:.3f} Hz")
    c2.metric("Calculated Gravity ($g$)", f"{calc_g:.2f} m/s²")
    c3.metric("Result in G-units", f"{g_ratio:.2f} g", delta=f"{percent_error:.1f}% Error", delta_color="inverse")

    if percent_error < 10.0:
        st.success(f"✅ **Model Calibrated:** Your gait closely follows the inverted pendulum model for Earth gravity.")
    elif g_ratio > 1.2:
        st.warning(f"⚠️ **High Gravity Result ({g_ratio:.2f} g):** Your walking frequency is high for your leg length.")
        st.info("""
        **Troubleshooting:** * Did you use the **Step** frequency instead of the **Stride** frequency? (Stride = 2 steps).
        * Were you walking with very 'stiff' legs? This increases the effective restoring force.
        """)
    elif g_ratio < 0.8:
        st.error(f"⚠️ **Low Gravity Result ({g_ratio:.2f} g):** Your gait is extremely fluid or slow.")
        st.info("🔍 **Troubleshooting:** Check your anatomical measurements (Floor to Hip). If $L_{eff}$ is too small, your calculated $g$ will drop.")
    else:
        st.info(f"📊 **Analysis Complete:** Your walk yielded {g_ratio:.2f} g. Review the Froude Number constraints to see if the model holds.")

    # --- 7. UI: Visualization ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frequencies, magnitude_norm, color='red', alpha=0.5, label='Normalized FFT Data')
    x_curve = np.linspace(0.5, upper_limit, 1000)
    y_curve = lorentzian(x_curve, *popt)
    ax.plot(x_curve, y_curve, color='black', lw=2.5, label=f'Lorentzian Fit ($f_0$ = {f0_fit:.3f} Hz)')
    ax.fill_between(x_curve, y_curve, color='lightgray', alpha=0.5)
    ax.axvline(f0_fit, color='blue', linestyle='--', alpha=0.8)
    ax.set_xlim(0, upper_limit)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Magnitude")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Analysis Error: {e}")
    st.warning("Ensure anatomical inputs are non-zero and CSV data is valid.")
