import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import moondream as md
from huggingface_hub import hf_hub_download
from models.vision_language_model import VisionLanguageModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SmolVLM-256m
def smolVLM_setup():
    print("Preparing SmolVLM-256m:")
    smol_name = "HuggingFaceTB/SmolVLM-256m-instruct"
    smol_model = AutoModelForImageTextToText.from_pretrained(
        smol_name, 
        dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)
    print("Making processor:")
    smol_processor = AutoProcessor.from_pretrained(smol_name)
    print("Complete.")
    return (smol_model, smol_processor);

# Moondream-0.5b
def moondream_setup():
    print("preparing Moondream-0.5b:")
    model_path = "../../moondream-0_5b-int8.mf"
    moon_model = md.vl(model=model_path)
    print("Complete.")
    return moon_model

# NanoVLM
def nanoVLM_setup():
    print("Preparing NanoVLM-222M:")
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M")
    print("Complete.")
    return model

print("VLM BENCHMARKING SCRIPT")
print(f"Running on device {DEVICE}")

print("Setup for models")
(smol_model, smol_processor) = smolVLM_setup()
moon_model = moondream_setup()
nano_model = nanoVLM_setup()

print("TKTK: Setup for benchmarks (NOT IMPLEMENTED)")
# using DocLayNet as dataset
# tasks related to: visual question answering, optical character recognition
# collect accuracy metrics, measure latency and amount of memory used for each task

print("TKTK: Benchmarking models (NOT IMPLEMENTED)")

print("TKTK: Generating output (NOT IMPLEMENTED)")

print("DONE!")