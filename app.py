


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


# ================= PAGE CONFIG =================
st.set_page_config(page_title="Speaker Diarization", layout="wide")

# ================= TAB VISIBILITY CSS =================
st.markdown("""
<style>
button[data-baseweb="tab"] {
    font-size: 18px;
    font-weight: 600;
    padding: 10px 25px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)


# ================= HEADER =================
st.markdown("""
<div style="text-align:center; padding:15px">
<h1>Multilingual Speaker Diarization Platform</h1>
<p style="color:gray">
Fine-tuned AI model for Hindi speech segmentation and speaker tracking
</p>
</div>
""", unsafe_allow_html=True)


# ================= SIDEBAR =================
with st.sidebar:
    st.title("Controls")
    min_spk = st.number_input("Min Speakers", 1, 10, 1)
    max_spk = st.number_input("Max Speakers", 1, 10, 5)

    st.markdown("---")
    st.caption("Model: Fine-tuned PyAnnote")
    st.caption("Audio: 16 kHz mono")


# ================= LOAD PIPELINE =================
@st.cache_resource
def get_pipeline():
    return load_diarization_pipeline()


pipeline = get_pipeline()


# ================= INPUT MODE =================
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

    st.info("Press start, speak, and stop recording")

    audio = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        key="recorder"
    )

    if audio:
        raw_path = "mic_recording.wav"
        with open(raw_path, "wb") as f:
            f.write(audio["bytes"])

        st.success("Recording captured")


# ==========================================================
# ================= RUN PIPELINE ===========================
# ==========================================================
if raw_path is not None:

    processed_path = raw_path.replace(".wav", "_16k.wav")
    convert_audio_to_16k_mono(raw_path, processed_path)

    st.divider()
    st.subheader("Input Audio")
    st.audio(processed_path)

    with st.spinner("Analyzing speakers..."):
        diarization = pipeline(
            processed_path,
            min_speakers=min_spk,
            max_speakers=max_spk
        )

    st.success("Analysis complete")


    # ======================================================
    # ================= PREPARE DATA =======================
    # ======================================================
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


    # ======================================================
    # ================= SUMMARY CARDS ======================
    # ======================================================
    st.divider()
    st.subheader("Session Summary")

    total_duration = len(y) / sr
    num_speakers = len(set([s[2] for s in segments_for_plot]))
    num_segments = len(segments_for_plot)

    lengths = [(end - start) for start, end, _ in segments_for_plot]
    avg_len = np.mean(lengths) if lengths else 0
    max_len = np.max(lengths) if lengths else 0

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        with st.container(border=True):
            st.metric("Total Duration", f"{total_duration:.1f} sec")

    with c2:
        with st.container(border=True):
            st.metric("Speakers", num_speakers)

    with c3:
        with st.container(border=True):
            st.metric("Speech Turns", num_segments)

    with c4:
        with st.container(border=True):
            st.metric("Average Segment", f"{avg_len:.2f} sec")

    with c5:
        with st.container(border=True):
            st.metric("Longest Segment", f"{max_len:.2f} sec")


    # ======================================================
    # ================= TABS ===============================
    # ======================================================
    st.divider()
    st.markdown("### Results Navigation")

    tab1, tab2, tab3 = st.tabs(["Segments", "Timeline", "Accuracy"])


    # ======================================================
    # ================= SEGMENTS TAB =======================
    # ======================================================
    with tab1:
        st.subheader("Detected Speaker Segments")

        cols_per_row = 2  # change to 3 if you want more compact
        rows = [segments_for_plot[i:i + cols_per_row]
                for i in range(0, len(segments_for_plot), cols_per_row)]

        segment_index = 0

        for row in rows:
            columns = st.columns(cols_per_row)

            for col, (start, end, label) in zip(columns, row):

                start_sample = int(start * sr)
                end_sample = int(end * sr)
                segment_audio = y[start_sample:end_sample]

                segment_path = f"segment_{segment_index}.wav"
                sf.write(segment_path, segment_audio, sr)

                with col:
                    with st.container(border=True):
                        st.markdown(f"**{segment_index+1}. {label}**")
                        st.caption(f"{start:.2f}s - {end:.2f}s")
                        st.audio(segment_path)

                segment_index += 1

    # ======================================================
    # ================= TIMELINE TAB =======================
    # ======================================================
    with tab2:
        st.subheader("Speaker Activity Timeline")

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
    # ================= DER TAB ============================
    # ======================================================
    with tab3:
        st.subheader("Diarization Error Rate")

        st.markdown("""
        **Formula**

        DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Reference Time
        """)

        if mode == " Upload WAV":
            filename = os.path.splitext(uploaded_file.name)[0]
            rttm_path = os.path.join("dataset", "rttm", filename + ".rttm")

            if os.path.exists(rttm_path):
                try:
                    reference = load_rttm(rttm_path)[filename]
                    metric = DiarizationErrorRate()

                    der_value = metric(reference, diarization)
                    components = metric.compute_components(reference, diarization)

                    with st.container(border=True):
                        c1, c2 = st.columns(2)

                        with c1:
                            st.metric("DER", f"{der_value:.3f}")
                            st.metric("DER (%)", f"{der_value*100:.2f}")

                        with c2:
                            st.write("Breakdown")
                            st.write(f"Missed Speech: {components['missed detection']:.2f}")
                            st.write(f"False Alarm: {components['false alarm']:.2f}")
                            st.write(f"Confusion: {components['confusion']:.2f}")

                except Exception as e:
                    st.error(f"DER calculation failed: {e}")

            else:
                st.warning("Reference RTTM not available.")
        else:
            st.info("DER not available for live recording.")


    # ======================================================
    # ================= DOWNLOAD ===========================
    # ======================================================
    st.divider()
    st.subheader("Export")

    try:
        rttm_text = diarization.to_rttm()
        st.download_button(
            "Download RTTM",
            rttm_text,
            file_name="diarization_output.rttm"
        )
    except:
        pass


# ================= FOOTER =================
st.markdown("""
<hr>
<p style='text-align:center; color:gray'>
Developed during CDAC Internship • AI Speaker Diarization System
</p>
""", unsafe_allow_html=True)
