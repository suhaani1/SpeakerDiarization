import streamlit as st
import tempfile
import os
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from diarization_engine import load_diarization_pipeline
from audio_utils import convert_audio_to_16k_mono

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from streamlit_mic_recorder import mic_recorder


# ================= PAGE =================
st.set_page_config(page_title="Hindi Speaker Diarization", layout="wide")

st.markdown("""
<h1 style='text-align:center;'> Multilingual Speaker Diarization Platform</h1>
<p style='text-align:center; color:gray;'>
Fine-tuned AI model for Hindi speech segmentation
</p>
""", unsafe_allow_html=True)


# ================= LOAD PIPELINE =================
@st.cache_resource
def get_pipeline():
    return load_diarization_pipeline()


pipeline = get_pipeline()

# ================= MODE =================
st.divider()
mode = st.radio(
    "Choose Input Source",
    [" Upload WAV", " Record from Mic"],
    horizontal=True
)

raw_path = None
uploaded_file = None


# ==========================================================
# ===================== UPLOAD MODE ========================
# ==========================================================
if mode == " Upload WAV":

    uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as raw:
            raw.write(uploaded_file.read())
            raw_path = raw.name


# ==========================================================
# ===================== MIC MODE ===========================
# ==========================================================
elif mode == " Record from Mic":

    st.info("Press start → speak → stop")

    audio = mic_recorder(
        start_prompt="⏺ Start Recording",
        stop_prompt="⏹ Stop Recording",
        key="recorder"
    )

    if audio:
        raw_path = "mic_recording.wav"
        with open(raw_path, "wb") as f:
            f.write(audio["bytes"])

        st.success("Recording captured!")


# ==========================================================
# ================= RUN PIPELINE ===========================
# ==========================================================
if raw_path is not None:

    processed_path = raw_path.replace(".wav", "_16k.wav")
    convert_audio_to_16k_mono(raw_path, processed_path)

    st.divider()
    st.subheader(" Input Audio")
    st.audio(processed_path)

    with st.spinner(" AI is analyzing speakers..."):
        diarization = pipeline(
            processed_path,
            min_speakers=1,
            max_speakers=5
        )

    st.success("Analysis complete!")

    # ======================================================
    # ================= DISPLAY SEGMENTS ===================
    # ======================================================
    st.divider()
    st.subheader(" Detected Speaker Segments")

    y, sr = librosa.load(processed_path, sr=16000)

    speaker_map = {}
    counter = 1

    segments_for_plot = []

    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):

        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {counter}"
            counter += 1

        label = speaker_map[speaker]

        segments_for_plot.append((turn.start, turn.end, label))

        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown(f"###  {label}")
            st.caption(f"{turn.start:.2f}s → {turn.end:.2f}s")

        with col2:
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            segment_audio = y[start_sample:end_sample]

            segment_path = f"segment_{i}.wav"
            sf.write(segment_path, segment_audio, sr)
            st.audio(segment_path)

        st.divider()

    # ======================================================
    # ================= SPEAKER GRAPH ======================
    # ======================================================
    st.subheader(" Speaker Activity Timeline")

    speakers = sorted(set([s[2] for s in segments_for_plot]))
    speaker_to_y = {spk: i for i, spk in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(12, len(speakers) * 1.2 + 1))

    colors = plt.cm.get_cmap("tab10", len(speakers))

    for start, end, spk in segments_for_plot:
        y_pos = speaker_to_y[spk]
        ax.barh(
            y_pos,
            end - start,
            left=start,
            height=0.5,
            color=colors(y_pos)
        )

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Who spoke when")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    st.pyplot(fig)

    # ======================================================
    # ================= DER ================================
    # ======================================================
    st.subheader(" Accuracy (DER)")

    if mode == " Upload WAV":
        filename = os.path.splitext(uploaded_file.name)[0]
        rttm_path = os.path.join("dataset", "rttm", filename + ".rttm")

        if os.path.exists(rttm_path):
            try:
                reference = load_rttm(rttm_path)[filename]
                metric = DiarizationErrorRate()
                der_value = metric(reference, diarization)

                st.success(f"DER: {der_value:.3f}  ({der_value*100:.1f}%)")

            except Exception as e:
                st.error(f"DER calculation failed: {e}")

        else:
            st.warning("Reference RTTM not available.")

    else:
        st.info("DER not available for live recording.")
