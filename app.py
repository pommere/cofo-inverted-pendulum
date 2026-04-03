import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from PIL import Image
import io
import os
import zipfile  # New addition for handling zip uploads

# --- 1. SETTINGS & BRANDING ---
logo_path = "cofo-logo.jpg"

# Try to load favicon for the tab
try:
    favicon = Image.open(logo_path)
    st.set_page_config(
        page_title="CofO | Inverted Pendulum Lab", 
        page_icon=favicon, 
        layout="centered"
    )
except:
    st.set_page_config(
        page_title="CofO | Inverted Pendulum Lab", 
        layout="centered"
    )

st.markdown("""
    <style>
        section[data-testid="stSidebar"] * {
            color: white !important;
        }
        section[data-testid="stSidebar"] input {
            color: #8D203C !important; 
        }
        .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp span {
            color: #000000;
        }
        [data-testid="stMetricLabel"] {
            color: #444444 !important;
        }
        [data-testid="stMetricValue"] {
            color: #000000 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-top: 1px solid #ffffff44 !important;
        }
    </style>
""", unsafe_allow_html=True)

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_container_width=True)
else:
    st.sidebar.warning(f"Logo '{logo_path}' not found.")

st.sidebar.markdown("### **College of the Ozarks**\nDepartment of Mathematics and Physics")
st.sidebar.divider()

col1, col2 = st.columns([1, 4]) 

with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=128) 

with col2:
    st.markdown(f"""
        <h1 style='color: #8D203C; margin-bottom: 0; padding-top: 10px;'>Inverted Pendulum Lab</h1>
        <p style='color: #002147; font-style: italic; font-size: 1.5em; margin-top: 0;'>
        College of the Ozarks | "Hard Work U"
        </p>
    """, unsafe_allow_html=True)

st.markdown(r"""
Welcome to the Physics Lab! Students deduce the local acceleration due to gravity ($g$)
by modeling human locomotion as an **inverted pendulum**. 

1. Upload your **Phyphox CSV** or **ZIP** file below.
2. The app will calculate the FFT to estimate $g$ from your stride period.
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
    L_eff = 0.55 * L_total 
    stride_period = 2 / step_freq
    g_calc = (4 * np.pi**2 * L_eff) / (stride_period**2)
    return g_calc, L_eff

# --- 4. File Upload & Processing ---
uploaded_file = st.file_uploader("Upload your Phyphox Data", type=["csv", "zip"])

df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file) as z:
                # Find file containing "Raw Data.csv" anywhere in the path
                target_file = next((f for f in z.namelist() if "Raw Data.csv" in f), None)
                if target_file:
                    with z.open(target_file) as f:
                        df = pd.read_csv(f)
                    st.success(f"Successfully extracted: {target_file}")
                else:
                    st.error("Could not find 'Raw Data.csv' inside the ZIP archive.")
                    st.stop()
        else:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded your CSV data.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    # Synthetic Data Fallback
    t = np.linspace(0, 10, 1000)
    synth_accel = 9.8 + 2.0 * np.sin(2 * np.pi * 1.475 * t) + np.random.normal(0, 0.3, 1000)
    df = pd.DataFrame({'Time (s)': t, 'Absolute acceleration (m/s^2)': synth_accel})
    st.info("💡 **No file uploaded yet.** Displaying synthetic 'Perfect Walk' data (1.475 Hz).")

# --- 5. Data Analysis & Physics ---
try:
    time = df['Time (s)'].values
    accel = df['Absolute acceleration (m/s^2)'].values

    dt = np.mean(np.diff(time))
    accel_detrended = accel - np.mean(accel)
    fft_values = np.fft.rfft(accel_detrended)
    frequencies = np.fft.rfftfreq(len(accel), d=dt)
    magnitude_norm = np.abs(fft_values) / len(accel)
    magnitude_norm /= (np.max(magnitude_norm) if np.max(magnitude_norm) != 0 else 1)

    search_mask = (frequencies >= 1.0) & (frequencies <= 4.0)
    f_search = frequencies[search_mask]
    m_search = magnitude_norm[search_mask]
    peaks, _ = find_peaks(m_search, height=0.1, distance=20)
    f_step_guess = f_search[peaks[0]] if len(peaks) > 0 else f_search[np.argmax(m_search)]

    upper_limit = max(3.0, 1.5 * f_step_guess)
    mask = (frequencies >= 0.5) & (frequencies <= upper_limit)
    popt, _ = curve_fit(lorentzian, frequencies[mask], magnitude_norm[mask], p0=[1.0, f_step_guess, 0.1])
    f0_fit = popt[1]

    calc_g, L_eff = calculate_g_physics(f0_fit, h_hip, h_ankle)
    percent_error = abs(calc_g - local_g) / local_g * 100
    g_ratio = calc_g / local_g
    v_derived = (L_eff * f0_fit * np.pi) / 2
    froude_num = (v_derived**2) / (local_g * L_eff)

    # --- 6. UI: Results Display ---
    st.subheader("Lab Analysis Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Step Frequency ($f_0$)", f"{f0_fit:.3f} Hz")
    c2.metric("Calculated $g$", f"{calc_g:.2f} m/s²")
    c3.metric("G-units", f"{g_ratio:.2f} g", delta=f"{percent_error:.1f}% Error", delta_color="inverse")
    c4.metric("Froude Number", f"{froude_num:.2f}")

    if froude_num > 0.5:
        st.error(f"🏃 **Froude Limit Exceeded:** Your Fr is {froude_num:.2f}. The model is likely breaking down!")
    elif percent_error < 10.0:
        st.success(f"✅ **Model Calibrated:** Your gait closely follows the inverted pendulum model.")
    
    with st.expander("📊 Energy Exchange & Velocity Theory"):
        st.markdown(f"""
        Walking is a continuous exchange of energy ($E_{{total}} = KE + PE$). 
        By analyzing your step frequency ($f_0$) and effective leg length ($L_{{eff}}$), we derived a 
        **Calculated Velocity** of **{v_derived:.2f} m/s**.
        """)

    if g_ratio > 1.2:
        st.warning(f"⚠️ **High Gravity Result:** Troubleshooting: Did you use Step frequency instead of Stride?")
    elif g_ratio < 0.8:
        st.warning(f"⚠️ **Low Gravity Result:** Troubleshooting: Check anatomical measurements.")

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