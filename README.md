
# Speaker Diarization using Fine-Tuned PyAnnote Segmentation

## Project Overview

This project implements **speaker diarization**, answering the fundamental question:

**“Who spoke when?”**

The system is developed using PyAnnote Audio. A speaker segmentation model is fine-tuned on custom audio data, followed by speaker embedding extraction and clustering to identify different speakers within an audio recording.

This project is suitable for:

* Call analysis
* Meeting transcription
* Surveillance and forensics
* Conversational AI pipelines
* Academic research and viva evaluation

---

## What is Speaker Diarization?

Speaker diarization divides an audio file into:

* Speech vs. non-speech
* Speaker segments
* Speaker labels (Speaker 1, Speaker 2, etc.)

**Example output**

```
00:00–00:12  → Speaker 1
00:12–00:20  → Speaker 2
00:20–00:35  → Speaker 1
```

---

## Project Architecture

```
Audio Input
   ↓
Preprocessing (Mono + 16 kHz)
   ↓
Fine-Tuned Segmentation Model (PyAnnote)
   ↓
Speech Activity Detection
   ↓
Speaker Embedding Extraction
   ↓
Agglomerative Clustering
   ↓
Final Speaker Diarization Output (RTTM / Timeline)
```

---

## Technologies and Libraries Used

* Python 3.9+
* PyTorch
* PyAnnote Audio
* Torchaudio
* Scikit-learn
* NumPy
* Matplotlib
* RTTM format for evaluation

---

## Project Structure

```
├── dataset/
│   ├── audio/                 # Training & evaluation audio files
│   ├── rttm/                  # Ground truth RTTM files
│   └── database.yml           # PyAnnote database configuration
│
├── models/
│   └── segmentation/          # Fine-tuned segmentation checkpoints
│
├── scripts/
│   ├── train_segmentation.py  # Model fine-tuning script
│   ├── diarization.py         # Inference + clustering
│   └── visualize.py           # Diarization visualization
│
├── outputs/
│   ├── rttm/                  # Predicted RTTM files
│   └── plots/                 # Speaker timeline plots
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Key Features

* Fine-tuned speaker segmentation model
* Works on real-world conversational audio
* Supports a variable number of speakers
* Uses Agglomerative Clustering
* Produces RTTM diarization output
* Visualization of speaker segments

---

## Audio Requirements (Important)

All audio must be:

* Mono (single channel)
* 16 kHz sampling rate
* WAV format

**Reason:**
PyAnnote models are trained on mono 16 kHz audio. Different formats or stereo input may reduce accuracy.

---

## How to Run the Project

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

* Place audio files in `dataset/audio/`
* Place RTTM files in `dataset/rttm/`
* Configure paths in `dataset/database.yml`

### 4. Fine-Tune Segmentation Model

```bash
python scripts/train_segmentation.py
```

### 5. Run Speaker Diarization

```bash
python scripts/diarization.py --audio sample.wav
```

### 6. Visualize Output

```bash
python scripts/visualize.py
```

---

## Output Formats

* RTTM files for evaluation
* Speaker timeline plots
* Segment-level speaker annotations

---

## Evaluation Metrics

* Diarization Error Rate (DER)
* False Alarm
* Missed Speech
* Speaker Confusion

---

## Model Details

* Base Model: PyAnnote speaker segmentation
* Training: Supervised fine-tuning
* Loss Function: Binary cross-entropy (segmentation)
* Clustering: Agglomerative hierarchical clustering
* Embeddings: Speaker representations extracted from the PyAnnote pipeline

---

## Limitations

* Overlapping speech remains challenging
* Performance depends on audio quality
* Requires sufficient labeled RTTM data

---

## Future Improvements

* Overlap-aware diarization
* Domain-specific embedding fine-tuning
* Real-time diarization
* Integration with ASR (speech-to-text)

---


## License

This project is intended for academic and learning purposes.
Pre-trained models belong to their respective authors.

---

## Acknowledgements

* PyAnnote Audio Team
* HuggingFace
* PyTorch Community


