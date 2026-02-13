import os
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline, Model


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def load_diarization_pipeline():

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )

    # Optional fine-tuned model
    model_path = "models\hindi_segmentation.pt"

    if os.path.exists(model_path):
        segmentation_model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=HF_TOKEN
        )

        state_dict = torch.load(model_path, map_location="cpu")
        segmentation_model.load_state_dict(state_dict)
        segmentation_model.eval()

        pipeline.segmentation.model = segmentation_model
        print("✅ Using Hindi fine-tuned model")

    else:
        print("⚠ Using default pretrained model")

    return pipeline
