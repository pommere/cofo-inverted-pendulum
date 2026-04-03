import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from PIL import Image
import io
import os
import zipfile

# --- 1. SETTINGS & BRANDING ---
logo_path = "cofo-logo.jpg"

try:
    favicon = Image.open(logo_path)
    st.set_page_config(page_title="CofO | Biokinematic Pendulum", page_icon=favicon, layout="centered")
except:
    st.set_page_config(page_title="CofO | Biokinematic Pendulum", layout="centered")

st.markdown("""
    <style>
        section[data-testid="stSidebar"] * { color: white !important; }
        section[data-testid="stSidebar"] input { color: #8D203C !important; }
        .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp span { color: #000000; }
        [data-testid="stMetricLabel"] { color: #444444 !important; }
        [data-testid="stMetricValue"] { color: #000000 !important; }
        section[data-testid="stSidebar"] hr { border-top: 1px solid #ffffff44 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MAIN HEADER ---
col1, col2 = st.columns([1, 4]) 
with col1:
    if os.path.exists(logo_path): st.image(logo_path, width=120) 

with col2:
    st.markdown(f"""
        <h1 style='color: #8D203C; margin-bottom: 0;'>Biokinematic Pendulum Lab</h1>
        <p style='color: #002147; font-style: italic; font-size: 1.2em; margin-top: 0;'>
        Deducing Earth's Gravity via Human Locomotion
        </p>
    """, unsafe_allow_html=True)

# --- 3. INTRODUCTION (FLESHED OUT FROM MANUAL) ---
st.markdown(r"""
### **The Inverted Pendulum Model**
In biomechanics, human gait is modeled as an **Inverted Pendulum**. As you walk, your body trades **Kinetic Energy** ($KE$) for **Gravitational Potential Energy** ($PE$). 
At the apex of a step, your Center of Mass (CoM) is highest, and your velocity is lowest.

By treating each step as a **Monte Carlo sample**, this app aggregates hundreds of oscillations to isolate the resonant frequency ($f$) of your unique "biological pendulum." 

$$g = 4\pi^2 f^2 L_{CM}$$

**Safety First:** Ensure your device is secured firmly. Avoid looking at the screen while walking!
""")

# --- 4. SIDEBAR: ANATOMICAL MEASUREMENTS ---
st.sidebar.header("1. Anatomical Measurements")
st.sidebar.markdown("Measure from the floor to these points:")
h_hip = st.sidebar.number_input("Floor to Hip (Greater Trochanter) [cm]", value=90.0, help="The pivot point of your leg pendulum.")
h_ankle = st.sidebar.number_input("Floor to Ankle (Talus) [cm]", value=8.0)

st.sidebar.header("2. Environment")
local_g = st.sidebar.number_input("Accepted $g$ (m/s²)", value=9.806)

# --- 5. PHYSICS CALCULATIONS ---
def lorentzian(x, a, x0, gamma):
    return a * (gamma**2 / ((x - x0)**2 + gamma**2))

def calculate_physics_results(step_freq, hip_cm, ankle_cm):
    # L = distance from talus to greater trochanter
    L_meters = (hip_cm - ankle_cm) / 100.0 
    # Center of Mass adjustment (0.55L from manual)
    L_cm_eff = 0.55 * L_meters 
    
    # g = (2*pi*f)^2 * L_cm
    g_calc = (2 * np.pi * step_freq)**2 * L_cm_eff
    
    # Velocity derived from energy exchange theory
    v_derived = (L_cm_eff * step_freq * np.pi) / 2
    froude = (v_derived**2) / (9.806 * L_cm_eff)
    
    return g_calc, L_cm_eff, v_derived, froude

# --- 6. FILE UPLOAD & PROCESSING ---
uploaded_file = st.file_uploader("Upload your Phyphox 'Acceleration' Data (CSV or ZIP)", type=["csv", "zip"])
df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file) as z:
                target_file = next((f for f in z.namelist() if "Raw Data.csv" in f), None)
                if target_file:
                    with z.open(target_file) as f: df = pd.read_csv(f)
                    st.success(f"Extracted: {target_file}")
                else:
                    st.error("Missing 'Raw Data.csv' in ZIP."); st.stop()
        else:
            df = pd.read_csv(uploaded_file)
            st.success("CSV Loaded Successfully.")
    except Exception as e:
        st.error(f"Error: {e}"); st.stop()
else:
    # Synthetic Data (Trial 1 Baseline example)
    t = np.linspace(0, 30, 3000)
    synth_accel = 9.8 + 2.0 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.4, 3000)
    df = pd.DataFrame({'Time (s)': t, 'Absolute acceleration (m/s^2)': synth_accel})
    st.info("💡 **No data uploaded.** Showing a synthetic 1.5 Hz baseline walk.")

# --- 7. DATA ANALYSIS ---
try:
    time, accel = df['Time (s)'].values, df['Absolute acceleration (m/s^2)'].values
    duration = time[-1] - time[0]
    
    if duration < 10.0:
        st.error(f"⚠️ **Trial Too Short ({duration:.1f}s):** The manual recommends 20-45s for baseline trials to reduce statistical noise."); st.stop()

    # FFT Analysis
    dt = np.mean(np.diff(time))
    accel_detrended = accel - np.mean(accel)
    fft_values = np.fft.rfft(accel_detrended)
    freqs = np.fft.rfftfreq(len(accel), d=dt)
    mags = np.abs(fft_values) / len(accel)
    mags /= (np.max(mags) if np.max(mags) != 0 else 1)

    # Find peak in human gait range (1-4 Hz)
    mask_search = (freqs >= 1.0) & (freqs <= 4.0)
    peaks, _ = find_peaks(mags[mask_search], height=0.1, distance=20)
    f_guess = freqs[mask_search][peaks[0]] if len(peaks) > 0 else 1.5

    # Fit Lorentzian to get a precise center frequency
    mask_fit = (freqs >= 0.5) & (freqs <= 4.5)
    try:
        popt, _ = curve_fit(lorentzian, freqs[mask_fit], mags[mask_fit], p0=[1.0, f_guess, 0.1])
        f0 = popt[1]
    except:
        f0 = f_guess; popt = [1.0, f_guess, 0.1]

    # Calculate Results
    calc_g, L_eff, v_calc, fr = calculate_physics_results(f0, h_hip, h_ankle)
    err = abs(calc_g - local_g) / local_g * 100

    # --- 8. RESULTS DISPLAY ---
    st.subheader("Data Extraction & Biometrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Step Freq ($f$)", f"{f0:.3f} Hz")
    c2.metric("Eff. Length ($L_{CM}$)", f"{L_eff:.3f} m")
    c3.metric("Calculated $g$", f"{calc_g:.2f} m/s²")
    c4.metric("Error", f"{err:.1f}%", delta_color="inverse")

    # Froude Number Logic
    st.markdown(f"### Froude Number ($Fr$): **{fr:.3f}**")
    if fr < 0.05:
        st.warning("**Trial 3 (The Shuffler) detected.** Kinetic energy is insufficient to reach the potential energy apex.")
    elif 0.20 <= fr <= 0.30:
        st.success("**Optimal Gait detected.** This is your 'Preferred' gait where the model is most accurate.")
    elif fr >= 0.5:
        st.error("**Trial 4 (Power Walk) / Transition detected.** You are approaching the limit where walking becomes mechanically expensive.")

    # FFT Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, mags, color='#002147', alpha=0.4, label='Gait Signal (FFT)')
    ax.plot(freqs[mask_fit], lorentzian(freqs[mask_fit], *popt), color='#8D203C', lw=2, label='Resonant Fit')
    ax.set_xlim(0, 5); ax.set_ylim(0, 1.1)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Normalized Magnitude")
    ax.legend(); st.pyplot(fig)

    with st.expander("📖 Synthesis Questions Help"):
        st.markdown(f"""
        * **Monte Carlo Sampling:** Your recording contains approximately **{int(f0 * duration)}** individual steps. 
        * **Velocity:** Based on your $Fr$ and $L_{{CM}}$, your estimated walking speed was **{v_calc:.2f} m/s**.
        * **Rigid vs. Soft Pendulum:** Note if your calculated $g$ is lower than $9.81$. Could knee flexion be shortening your effective $L$ during the swing?
        """)

except Exception as e:
    st.error(f"Analysis Failed: {e}")