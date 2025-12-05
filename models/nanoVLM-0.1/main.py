import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import moondream as md
from huggingface_hub import hf_hub_download
from datasets import load_dataset, DownloadConfig
from models.vision_language_model import VisionLanguageModel
import time
import psutil
import csv
import evaluate
import os

os.environ["HF_DATASETS_DOWNLOAD_TIMEOUT"] = "60000"  # seconds
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60000"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

# SmolVLM-256m
def smolVLM_setup():
    print("Preparing SmolVLM-256m:")
    smol_name = "HuggingFaceTB/SmolVLM-256m-instruct"
    smol_model = AutoModelForImageTextToText.from_pretrained(
        smol_name, 
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)
    print("Making processor:")
    smol_processor = AutoProcessor.from_pretrained(smol_name)
    print("Complete.")
    return (smol_model, smol_processor);

def smolVLM_run(model, processor, image, prompt):
    inputs = processor(image, prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Moondream-0.5b
def moondream_setup():
    print("preparing Moondream-0.5b:")
    model_path = "../../moondream-0_5b-int8.mf"
    moon_model = md.vl(model=model_path)
    print("Complete.")
    return moon_model

def moondream_run(model, image, prompt):
    return model.caption(image, prompt=prompt)

# NanoVLM
def nanoVLM_setup():
    print("Preparing NanoVLM-222M:")
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M")
    print("Complete.")
    return model

def nanoVLM_run(model, image, prompt):
    return model.generate(image, prompt)

# benchmarking functions
def measure_gpu_memory():
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0

def reset_gpu_memory():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

def measure_ram():
    return psutil.Process().memory_info().rss / 1024**2

def vqa_accuracy(prediction, ground_truth):
    return float(pred.strip().lower() == ground_truth.strip().lower())

def benchmark_model(model_name, run_func, dataset):
    results = {
        "ocr": {"cer": [], "wer": [], "latency": [], "gpu_mem": [], "ram": []},
        "vqa": {"accuracy": [], "latency": [], "gpu_mem": [], "ram": []},
    }

    # I want progress updates printed to the console.
    length = len(dataset)
    last_check = 0

    for i, sample in enumerate(dataset):
        
        image = sample["image"]
        ocr_ground_truth = sample["ocr_text"]
        question = sample["question"]
        answer = sample["answer"]

        # OCR test
        reset_gpu_memory()
        start_ram = measure_ram()
        start = time.time()

        ocr_prediction = run_func(image, "Extract all text from this document.")

        latency = time.time() - start
        gpu_mem = measure_gpu_memory()
        ram_used = measure_ram() - start_ram
        
        results["ocr"]["latency"].append(latency)
        results["ocr"]["gpu_mem"].append(gpu_mem)
        results["ocr"]["ram"].append(ram_used)
        results["ocr"]["cer"].append(cer_metric.compute(predictions=[ocr_prediction], references=[ocr_ground_truth]))
        results["ocr"]["wer"].append(wer_metric.compute(predictions=[ocr_prediction], references=[ocr_ground_truth]))

        # VQA test
        reset_gpu_memory()
        start_ram = measure_ram()
        start = time.time()

        vqa_prediction = run_func(image, question)

        latency = time.time() - start
        gpu_mem = measure_gpu_memory()
        ram_used = measure_ram() - start_ram

        results["vqa"]["latency"].append(latency)
        results["vqa"]["gpu_mem"].append(gpu_mem)
        results["vqa"]["ram"].append(ram_used)
        results["vqa"]["accuracy"].append(vqa_accuracy(vqa_prediction, answer))

        # progress update print
        completion = 100 * i / length
        if completion - last_check >= 10:
            print(f"Completion: {100 * i / length}%")
            last_check = completion

    return results


print("VLM BENCHMARKING SCRIPT")
print(f"Running on device {DEVICE}")

print("Setup for models")
(smol_model, smol_processor) = smolVLM_setup()
moon_model = moondream_setup()
nano_model = nanoVLM_setup()

print("TKTK: Setup for benchmarks")
# using DocLayNet as dataset
# tasks related to: visual question answering, optical character recognition
# collect accuracy metrics, measure latency and amount of memory used for each task

raw_data = load_dataset("ds4sd/DocLayNet", split="test", download_config=DownloadConfig(num_proc=1, max_retries=20))


def build_benchmark(sample):
    image = sample["image"]

    ocr_text = "\n".join([i["text"] for i in sample["ocr"]])

    vqa_question = "How many text blocks are present in this document?"
    vqa_answer = srt(len(sample["ocr"]))

    return {
        "image": image,
        "ocr_text": ocr_text,
        "question": question,
        "answer": answer
    }

benchmark_dataset = raw.map(build_benchmark)

print("TKTK: Benchmarking models")

print("TESTING SMOLVLM:")
smol_results = benchmark_model(
    "smolVLM",
    lambda img, prompt: smolVLM_run(smol_model, smol_processor, img, prompt),
    benchmark_dataset
)

print("")
moondream_results = benchmark_model(
    "moondream",
    lambda img, prompt: moondream_run(moon_model, img, prompt),
    benchmark_dataset
)

nano_results = benchmark_model(
    "nanoVLM",
    lambda img, prompt: nanoVLM_run(nano_model, img, prompt),
    benchmark_dataset
)

print("TKTK: Generating output (NOT IMPLEMENTED)")

# TODO this is a placeholder function
# we should probably be writing data to a CSV or something instead of this so we can use the data elsewhere

def summarize(name, results):
    print(f"=== {name} ===")
    for task in ["ocr", "vqa"]:
        print(f"\n[{task.upper()}]")
        for key, values in results[task].items():
            print(f"{key}: {np.mean(values):.4f}")

summarize("SmolVLM-256m", smol_results)
summarize("Moondream-0.5b", moondream_results)
summarize("NanoVLM", nano_results)


print("DONE!")