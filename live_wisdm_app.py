import requests
import time
import numpy as np
import joblib
from collections import deque
from scipy.stats import iqr
from scipy.fft import rfft, rfftfreq

# ==============================
# 1. CONFIG
# ==============================

# Put the EXACT remote access URL from Phyphox here + ?acceleration
# Example: "http://10.1.234.125/get?acceleration"
PHONE_URL = "http://172.20.10.1/get?accX&accY&accZ"

WINDOW_SIZE = 128          # MUST match the window size you used for training
SAMPLING_RATE = 20         # WISDM is ~20 Hz, ok for FFT

# ==============================
# 2. LOAD MODEL + SCALER + ENCODER
# ==============================

rf_model = joblib.load("wisdm_rf_model.joblib")
scaler = joblib.load("wisdm_scaler.joblib")
label_encoder = joblib.load("wisdm_label_encoder.joblib")
class_names = label_encoder.classes_

print("Loaded model, scaler, and label encoder.")
print("Classes:", class_names)

# ==============================
# 3. FEATURE EXTRACTION
# (MUST MATCH NOTEBOOK VERSION)
# ==============================

def extract_features(window, sampling_rate=SAMPLING_RATE):
    """
    window: numpy array of shape (WINDOW_SIZE, 3) -> columns: x, y, z
    returns: feature vector of length 31 (same as in training)
    """
    x = window[:, 0]
    y = window[:, 1]
    z = window[:, 2]

    feats = []

    def axis_features(axis):
        f = []
        # basic stats
        f.append(axis.mean())
        f.append(axis.std())
        f.append(axis.min())
        f.append(axis.max())
        f.append(np.median(axis))
        # IQR
        f.append(iqr(axis))
        # energy
        f.append(np.sum(axis**2) / len(axis))
        # zero crossings
        f.append(np.sum(np.diff(np.sign(axis)) != 0))
        return f

    # time-domain per axis
    for axis in [x, y, z]:
        feats.extend(axis_features(axis))

    # magnitude features
    mag = np.sqrt(x * x + y * y + z * z)
    feats.append(mag.mean())
    feats.append(mag.std())
    feats.append(iqr(mag))
    feats.append(np.sum(mag**2) / len(mag))
    feats.append(np.sum(np.diff(np.sign(mag)) != 0))

    # simple frequency-domain on magnitude
    fft_vals = rfft(mag)
    fft_freqs = rfftfreq(len(mag), d=1 / sampling_rate)
    fft_power = np.abs(fft_vals)

    # dominant frequency
    dom_idx = np.argmax(fft_power)
    dom_freq = fft_freqs[dom_idx]
    feats.append(dom_freq)

    # frequency-band energy
    feats.append(np.sum(fft_power**2))

    feats = np.array(feats, dtype=float)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats

# ==============================
# 4. LIVE DATA LOOP
# ==============================

acc_buffer = deque(maxlen=WINDOW_SIZE)
last_activity = None

print("\nListening to live Phyphox data from:")
print(PHONE_URL)
print("Make sure Phyphox 'Acceleration' experiment is running with remote access enabled.\n")

while True:
        try:
            r = requests.get(PHONE_URL, timeout=1)
            data = r.json()

            buf = data.get("buffer", {})
            if not buf:
                print("Empty buffer dict, keys:", list(data.keys()))
                time.sleep(0.2)
                continue

        # Expecting accX, accY, accZ in buffer
            if not all(k in buf for k in ("accX", "accY", "accZ")):
                print("Acceleration buffers not found, got buffer keys:", list(buf.keys()))
                time.sleep(0.2)
                continue

            x_buf = buf["accX"].get("buffer", [])
            y_buf = buf["accY"].get("buffer", [])
            z_buf = buf["accZ"].get("buffer", [])

            # Need at least one value in each
            if not x_buf or not y_buf or not z_buf:
                # no new samples yet
                time.sleep(0.05)
                continue

            # Take the latest sample from each axis
            ax = x_buf[-1]
            ay = y_buf[-1]
            az = z_buf[-1]

            acc_buffer.append((ax, ay, az))

            if len(acc_buffer) == WINDOW_SIZE:
                window_arr = np.array(acc_buffer)  # shape: (WINDOW_SIZE, 3)
                feat_vec = extract_features(window_arr).reshape(1, -1)
                feat_scaled = scaler.transform(feat_vec)
                pred_idx = rf_model.predict(feat_scaled)[0]
                activity = class_names[pred_idx]

                if activity != last_activity:
                    print("Predicted activity:", activity)
                    last_activity = activity

            time.sleep(0.02)

        except Exception as e:
            print("Error:", e)
            time.sleep(0.5)
