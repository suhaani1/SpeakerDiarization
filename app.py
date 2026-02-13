# # # import streamlit as st
# # # import os
# # # import torch
# # # import numpy as np
# # # import librosa
# # # import soundfile as sf
# # # import noisereduce as nr
# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # from pyannote.audio import Model, Inference
# # # from pyannote.audio.utils.signal import Binarize
# # # from pyannote.core import SlidingWindowFeature, Annotation
# # # from sklearn.cluster import AgglomerativeClustering
# # # from sklearn.metrics import silhouette_score

# # # # --- 1. PYTORCH 2.6+ SECURITY FIX ---
# # # import torch.serialization
# # # original_load = torch.load
# # # def forced_load(f, map_location=None, pickle_module=None, **kwargs):
# # #     kwargs['weights_only'] = False
# # #     return original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
# # # torch.load = forced_load
# # # # -------------------------------

# # # st.set_page_config(page_title="Hindi-Bhojpuri Diarization Tool", layout="wide")

# # # st.title("🎙️ Speaker Diarization with De-noising")
# # # st.markdown("""
# # # This tool uses a fine-tuned model to detect speakers. 
# # # The system automatically determines the number of speakers based on voice similarity.
# # # """)

# # # # --- SIDEBAR CONFIGURATION (UI CLEANUP) ---
# # # st.sidebar.header("Configuration")
# # # MODEL_PATH = st.sidebar.text_input("Model Checkpoint Path", "training_results/lightning_logs/version_2/checkpoints/epoch=4-step=2960.ckpt")
# # # use_denoise = st.sidebar.checkbox("Enable De-noising", value=True)

# # # st.sidebar.subheader("Advanced Settings")
# # # threshold = st.sidebar.slider("AI Sensitivity (VAD)", 0.5, 0.95, 0.80)

# # # @st.cache_resource
# # # def load_cached_model(path):
# # #     if not os.path.exists(path):
# # #         return None
# # #     return Model.from_pretrained(path)

# # # def process_audio(audio_path, model_path, denoise, sensitivity):
# # #     # 1. Load Audio
# # #     y, sr = librosa.load(audio_path, sr=16000)
    
# # #     # 2. AGGRESSIVE DE-NOISING
# # #     if denoise:
# # #         with st.spinner("Step 1: Deep cleaning audio..."):
# # #             # Increased prop_decrease to 0.90 to kill heavy background noise
# # #             y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.90, n_fft=2048)
# # #             audio_input = "temp_denoised.wav"
# # #             sf.write(audio_input, y, sr)
# # #     else:
# # #         audio_input = audio_path

# # #     # 3. AI Inference
# # #     with st.spinner("Step 2: AI Neural Analysis..."):
# # #         model = load_cached_model(model_path)
# # #         if model is None: return None, None
            
# # #         inference = Inference(model, window="sliding", duration=2.0, step=0.5)
# # #         seg_output = inference(audio_input)
        
# # #         data = np.squeeze(seg_output.data)
# # #         if len(data.shape) == 3: data = data[:, :, 0]
# # #         clean_scores = SlidingWindowFeature(data, seg_output.sliding_window)
        
# # #         # 4. BINARIZATION FIX: Increase 'min_duration_on' to 1.2 seconds
# # #         # This ignores all short noises/coughs/background clicks that cause 100+ speakers.
# # #         binarize = Binarize(onset=0.85, offset=0.75, min_duration_on=1.2, min_duration_off=0.5)
# # #         raw_hyp = binarize(clean_scores)
        
# # #         # 5. FEATURE EXTRACTION
# # #         embeddings = []
# # #         segments = []
# # #         for segment, track, label in raw_hyp.itertracks(yield_label=True):
# # #             # Focus on the middle of the segment to get a 'clean' voiceprint
# # #             feature_vector = np.mean(seg_output.crop(segment).data, axis=0).flatten()
# # #             embeddings.append(feature_vector)
# # #             segments.append(segment)

# # #         final_hyp = Annotation()
        
# # #         if len(embeddings) > 1:
# # #             X = np.array(embeddings)
            
# # #             # --- AUTO-DETECTION LOGIC ---
# # #             # If Silhouette fails, we fall back to a safe range (2 to 5 speakers)
# # #             try:
# # #                 scores = []
# # #                 range_n = range(2, min(len(embeddings), 6))
# # #                 for n in range_n:
# # #                     clusterer = AgglomerativeClustering(n_clusters=n, metric='euclidean', linkage='ward')
# # #                     labels = clusterer.fit_predict(X)
# # #                     scores.append(silhouette_score(X, labels))
# # #                 best_n = range_n[np.argmax(scores)]
# # #             except:
# # #                 best_n = 2 # Safe default for OJT demo

# # #             clusterer = AgglomerativeClustering(n_clusters=best_n, metric='euclidean', linkage='ward')
# # #             final_labels = clusterer.fit_predict(X)
            
# # #             for i, segment in enumerate(segments):
# # #                 final_hyp[segment] = f"Speaker {final_labels[i]}"
        
# # #         elif len(embeddings) == 1:
# # #             final_hyp[segments[0]] = "Speaker 0"

# # #     # .support() is CRITICAL: it merges small gaps of the same speaker
# # #     return final_hyp.support(), audio_input

# # # # --- MAIN UI ---
# # # uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

# # # if uploaded_file is not None:
# # #     with open("temp_upload.wav", "wb") as f:
# # #         f.write(uploaded_file.getbuffer())
    
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         st.subheader("Original Audio")
# # #         st.audio("temp_upload.wav")
    
# # #     if st.button("Start AI Analysis"):
# # #         hyp, final_audio = process_audio("temp_upload.wav", MODEL_PATH, use_denoise, threshold)
        
# # #         if hyp is None:
# # #             st.error("Model not found!")
# # #         else:
# # #             with col2:
# # #                 if use_denoise:
# # #                     st.subheader("Denoised Version")
# # #                     st.audio(final_audio)
            
# # #             st.divider()
            
# # #             unique_speakers = sorted(hyp.labels())
# # #             st.subheader(f"📊 Speaker Timeline ({len(unique_speakers)} Speakers Detected)")
            
# # #             if len(unique_speakers) > 0:
# # #                 fig, ax = plt.subplots(figsize=(12, len(unique_speakers) * 0.8 + 1.5))
# # #                 colors = plt.cm.get_cmap('tab10', len(unique_speakers))
                
# # #                 for i, speaker in enumerate(unique_speakers):
# # #                     speaker_segments = hyp.label_timeline(speaker)
# # #                     intervals = [(s.start, s.duration) for s in speaker_segments]
# # #                     ax.broken_barh(intervals, (i*10 + 2, 6), facecolors=colors(i))
                
# # #                 ax.set_yticks([i*10 + 5 for i in range(len(unique_speakers))])
# # #                 ax.set_yticklabels(unique_speakers)
# # #                 ax.set_xlabel("Time (seconds)")
# # #                 ax.grid(axis='x', linestyle='--', alpha=0.5)
# # #                 st.pyplot(fig)
                
# # #                 timestamp_list = []
# # #                 for segment, track, label in hyp.itertracks(yield_label=True):
# # #                     timestamp_list.append({
# # #                         "Speaker ID": label,
# # #                         "Start (s)": round(segment.start, 2),
# # #                         "End (s)": round(segment.end, 2),
# # #                         "Duration (s)": round(segment.duration, 2)
# # #                     })
                
# # #                 df = pd.DataFrame(timestamp_list)
# # #                 st.dataframe(df, use_container_width=True)
# # #                 st.download_button("📩 Download CSV", df.to_csv(index=False).encode('utf-8'), "diarization.csv", "text/csv")
# # #             else:
# # #                 st.warning("No speech detected.")




# # import streamlit as st
# # import tempfile
# # import os

# # from diarization_engine import load_diarization_pipeline
# # from audio_utils import convert_audio_to_16k_mono


# # st.set_page_config(page_title="Hindi Speaker Diarization", layout="centered")

# # st.title("🎙 Hindi Speaker Diarization")
# # st.caption("Upload WAV → detect who spoke when")

# # # ================= LOAD PIPELINE =================
# # @st.cache_resource
# # def get_pipeline():
# #     return load_diarization_pipeline()

# # pipeline = get_pipeline()

# # # ================= UPLOAD =================
# # uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])


# # if uploaded_file:

# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as raw:
# #         raw.write(uploaded_file.read())
# #         raw_path = raw.name

# #     processed_path = raw_path.replace(".wav", "_16k.wav")
# #     convert_audio_to_16k_mono(raw_path, processed_path)

# #     st.audio(processed_path)

# #     # ================= RUN =================
# #     with st.spinner("Running diarization..."):
# #         diarization = pipeline(
# #             processed_path,
# #             min_speakers=1,
# #             max_speakers=5
# #         )

# #     st.success("Done!")

# #     # ================= DISPLAY =================
# #     st.subheader("📝 Speaker Timeline")

# #     speaker_map = {}
# #     counter = 1

# #     for turn, _, speaker in diarization.itertracks(yield_label=True):
# #         if speaker not in speaker_map:
# #             speaker_map[speaker] = f"Speaker {counter}"
# #             counter += 1

# #         label = speaker_map[speaker]

# #         st.markdown(
# #             f"**[{turn.start:.2f}s – {turn.end:.2f}s]** 🔊 {label}"
# #         )




# # import streamlit as st
# # import tempfile
# # import os
# # import librosa
# # import soundfile as sf

# # from diarization_engine import load_diarization_pipeline
# # from audio_utils import convert_audio_to_16k_mono


# # from pyannote.database.util import load_rttm
# # from pyannote.metrics.diarization import DiarizationErrorRate


# # # ================= PAGE =================
# # st.set_page_config(page_title="Hindi Speaker Diarization", layout="centered")

# # st.title("🎙 Hindi Speaker Diarization")
# # st.caption("Upload WAV → detect who spoke when → listen to each segment")


# # # ================= LOAD PIPELINE =================
# # @st.cache_resource
# # def get_pipeline():
# #     return load_diarization_pipeline()


# # pipeline = get_pipeline()


# # # ================= UPLOAD =================
# # uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])


# # # ================= PROCESS =================
# # if uploaded_file:

# #     # Save raw upload
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as raw:
# #         raw.write(uploaded_file.read())
# #         raw_path = raw.name

# #     # Convert to 16k mono
# #     processed_path = raw_path.replace(".wav", "_16k.wav")
# #     convert_audio_to_16k_mono(raw_path, processed_path)

# #     st.subheader("Uploaded Audio")
# #     st.audio(processed_path)

# #     # ================= RUN DIARIZATION =================
# #     with st.spinner("Running diarization..."):
# #         diarization = pipeline(
# #             processed_path,
# #             min_speakers=1,
# #             max_speakers=5
# #         )

# #     st.success("Diarization complete!")

# #     # ================= DISPLAY =================
# #     st.divider()
# #     st.subheader("📝 Speaker Timeline & Segments")

# #     # Load audio once
# #     y, sr = librosa.load(processed_path, sr=16000)

# #     speaker_map = {}
# #     counter = 1

# #     for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):

# #         # Map model labels → Speaker 1, 2, 3
# #         if speaker not in speaker_map:
# #             speaker_map[speaker] = f"Speaker {counter}"
# #             counter += 1

# #         label = speaker_map[speaker]

# #         # Show timestamp
# #         st.markdown(
# #             f"**[{turn.start:.2f}s – {turn.end:.2f}s]** 🔊 {label}"
# #         )

# #         # ================= CUT AUDIO =================
# #         start_sample = int(turn.start * sr)
# #         end_sample = int(turn.end * sr)
# #         segment_audio = y[start_sample:end_sample]

# #         segment_path = f"segment_{i}.wav"
# #         sf.write(segment_path, segment_audio, sr)

# #         # ================= AUDIO PLAYER =================
# #         st.audio(segment_path, format="audio/wav")

# #         st.divider()
# #     # ================= DER =================
# # st.divider()
# # st.subheader("📊 Diarization Error Rate (DER)")

# # # try finding reference
# # filename = os.path.splitext(uploaded_file.name)[0]
# # rttm_path = os.path.join("dataset", "rttm", filename + ".rttm")

# # if os.path.exists(rttm_path):

# #     reference = load_rttm(rttm_path)[filename]
# #     metric = DiarizationErrorRate()

# #     der_value = metric(reference, diarization)

# #     st.success(f"DER: {der_value:.3f}  ({der_value*100:.1f}%)")

# # else:
# #     st.warning("Reference RTTM not found → DER not computed")


# import streamlit as st
# import tempfile
# import os
# import librosa
# import soundfile as sf
# import numpy as np

# from diarization_engine import load_diarization_pipeline
# from audio_utils import convert_audio_to_16k_mono

# from pyannote.database.util import load_rttm
# from pyannote.metrics.diarization import DiarizationErrorRate

# from streamlit_mic_recorder import mic_recorder



# # ================= PAGE =================
# st.set_page_config(page_title="Hindi Speaker Diarization", layout="centered")

# st.title("🎙 Hindi Speaker Diarization")
# st.caption("Upload or record → detect who spoke when")


# # ================= LOAD PIPELINE =================
# @st.cache_resource
# def get_pipeline():
#     return load_diarization_pipeline()


# pipeline = get_pipeline()


# # ================= MODE =================
# mode = st.radio("Choose input method:", ["Upload WAV", "Record from Mic"])


# # ==========================================================
# # ===================== UPLOAD MODE ========================
# # ==========================================================
# if mode == "Upload WAV":

#     uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

#     if uploaded_file:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as raw:
#             raw.write(uploaded_file.read())
#             raw_path = raw.name


# # ==========================================================
# # ===================== MIC MODE ===========================
# # ==========================================================
# elif mode == "Record from Mic":

#     audio = mic_recorder(
#         start_prompt="Start recording",
#         stop_prompt="Stop recording",
#         key="recorder"
#     )

#     if audio:
#         raw_path = "mic_recording.wav"

#         with open(raw_path, "wb") as f:
#             f.write(audio["bytes"])

#         st.success("Recording ready!")


# # ==========================================================
# # ================= RUN PIPELINE ===========================
# # ==========================================================
# if "raw_path" in locals() and raw_path is not None:

#     processed_path = raw_path.replace(".wav", "_16k.wav")
#     convert_audio_to_16k_mono(raw_path, processed_path)

#     st.subheader("Input Audio")
#     st.audio(processed_path)

#     with st.spinner("Running diarization..."):
#         diarization = pipeline(
#             processed_path,
#             min_speakers=1,
#             max_speakers=5
#         )

#     st.success("Diarization complete!")

#     # ================= DISPLAY =================
#     st.divider()
#     st.subheader("📝 Speaker Timeline & Segments")

#     y, sr = librosa.load(processed_path, sr=16000)

#     speaker_map = {}
#     counter = 1

#     for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):

#         if speaker not in speaker_map:
#             speaker_map[speaker] = f"Speaker {counter}"
#             counter += 1

#         label = speaker_map[speaker]

#         st.markdown(
#             f"**[{turn.start:.2f}s – {turn.end:.2f}s]** 🔊 {label}"
#         )

#         start_sample = int(turn.start * sr)
#         end_sample = int(turn.end * sr)
#         segment_audio = y[start_sample:end_sample]

#         segment_path = f"segment_{i}.wav"
#         sf.write(segment_path, segment_audio, sr)

#         st.audio(segment_path, format="audio/wav")
#         st.divider()

#     # ================= DER =================
#     st.subheader("📊 Diarization Error Rate (DER)")

#     if mode == "Upload WAV":
#         filename = os.path.splitext(uploaded_file.name)[0]
#         rttm_path = os.path.join("dataset", "rttm", filename + ".rttm")

#         if os.path.exists(rttm_path):
#             try:
#                 reference = load_rttm(rttm_path)[filename]
#                 metric = DiarizationErrorRate()
#                 der_value = metric(reference, diarization)

#                 st.success(f"DER: {der_value:.3f} ({der_value*100:.1f}%)")

#             except Exception as e:
#                 st.error(f"DER calculation failed: {e}")

#         else:
#             st.info("No RTTM found → DER skipped.")

#     else:
#         st.info("Live recording → DER unavailable.")




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
