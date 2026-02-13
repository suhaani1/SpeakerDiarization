
# Speaker Diarization with Fine-Tuned PyAnnote

## Introduction

Speaker diarization is the process of determining **who speaks at what time** in an audio recording.
This project builds a complete diarization pipeline by fine-tuning a PyAnnote segmentation model and combining it with speaker embeddings and clustering algorithms.

The system is designed for research, experimentation, and academic demonstrations, while remaining practical for real conversational data.

---

## Problem Statement

Given an input audio file containing multiple speakers, the system must:

* Detect where speech occurs
* Split the audio into homogeneous speaker regions
* Assign consistent speaker IDs across the timeline

---

## Where This Can Be Used

* Contact center and call monitoring
* Meeting analytics
* Forensic audio investigation
* Pre-processing for speech-to-text systems
* AI research projects and university evaluations

---

## Pipeline Overview

The diarization workflow follows a modular architecture:

```
Audio → Preprocess → Segmentation → Speech Regions
      → Embeddings → Clustering → Speaker Labels
```

### Step Description

1. **Preprocessing** – converts audio into mono, 16 kHz WAV.
2. **Segmentation** – neural model predicts speech boundaries.
3. **Embedding Extraction** – transforms speech chunks into speaker vectors.
4. **Clustering** – groups similar voices together.
5. **Output Generation** – produces RTTM files and timelines.

---

## Technical Stack

* Python
* PyTorch
* PyAnnote Audio
* Torchaudio
* Scikit-learn
* NumPy
* Matplotlib

---

## Repository Layout

```
dataset/        → audio, annotations, database config
models/         → trained checkpoints
scripts/        → training & inference programs
outputs/        → predictions and visualizations
```

---

## Capabilities

* Custom fine-tuning on labeled RTTM datasets
* Automatic handling of unknown number of speakers
* Works on long conversations
* Standard evaluation compatible with diarization benchmarks
* Visual inspection of speaker turns

---

## Input Audio Specification

For reliable results, audio must follow:

* **Single channel (mono)**
* **16,000 Hz sampling rate**
* **WAV format**

This matches the training conditions of most PyAnnote models.

---

---

## Training the Segmentation Model

```bash
python scripts/train_segmentation.py
```

The script uses dataset definitions from `database.yml` and stores checkpoints in the `models/` directory.

---

## Running Inference

```bash
python scripts/diarization.py --audio path/to/audio.wav
```

Predictions will be written to the `outputs/` folder.

---

## Visualization

```bash
python scripts/visualize.py
```

Generates speaker timelines for qualitative analysis.

---

## Generated Results

The system produces:

* RTTM diarization files
* Time-aligned speaker segments
* Graphical plots of speaker activity

---

## Performance Measurement

Evaluation is typically done using:

* Diarization Error Rate
* Missed speech
* False alarms
* Speaker confusion

---

## Design Choices

* Neural segmentation improves robustness over rule-based VAD.
* Embedding-based clustering allows flexible speaker counts.
* Hierarchical clustering provides interpretable grouping.

---

## Current Challenges

* Heavy overlap between speakers
* Background noise or reverberation
* Limited labeled data for adaptation

---

## Possible Extensions

* End-to-end neural diarization
* Online / streaming prediction
* Joint diarization + ASR
* Improved overlap modeling

---

## License Notice

Developed for academic and educational usage.
All third-party models remain property of their respective creators.

---

## Credits

Thanks to the open-source communities behind PyAnnote, PyTorch, and related research tools that make experimentation possible.



