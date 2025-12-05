import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
import moondream as md
from huggingface_hub import hf_hub_download
from datasets import load_dataset, DownloadConfig, Dataset
from models.vision_language_model import VisionLanguageModel
from data.processors import get_image_processor, get_tokenizer
import time
import psutil
import csv
import evaluate
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import json
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
import random
from collections import defaultdict

cells_dir = Path("../../DocLayNet/EXTRA/JSON")  # adjust path
os.environ["HF_DATASETS_DOWNLOAD_TIMEOUT"] = "60000"  # seconds
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60000"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def resize_for_smol(image, max_side=512):
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return image   # already small enough
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size)

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
    smol_processor.image_processor.size["longest_edge"] = 512
    print("Complete.")
    return (smol_model, smol_processor);

def smolVLM_run(model, processor, images, prompts, batch_size=4):
    results = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        for img, prompt in zip(batch_imgs, batch_prompts):
            messages = [[
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": p}
                    ]
                }]
                for p in batch_prompts
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(images=[img], text=prompt, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            decoded = processor.decode(out[0], skip_special_tokens=True)
            results.append(decoded)
        
    return results

# Moondream-0.5b
def moondream_setup():
    print("preparing Moondream-0.5b:")
    model_path = "../../moondream-0_5b-int8.mf"
    moon_model = md.vl(model=model_path)
    print("Complete.")
    return moon_model

def moondream_run(model, images, prompts, batch_size=4):
    results = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        
        for img, prompt in zip(batch_imgs, batch_prompts):
            results.append(model.query(img, prompt))
    return results

# NanoVLM
def nanoVLM_setup():
    print("Preparing NanoVLM-222M:")
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M")
    print("Complete.")
    return model



def nanoVLM_run(model, images, prompts, batch_size=4, max_new_tokens=20):
    image_processor = get_image_processor(model.cfg.vit_img_size)
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    results = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        img_tensors = torch.stack([image_processor(img) for img in batch_imgs]).to(DEVICE)
        tokenized = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        generated_ids = model.generate(
            input_ids=input_ids,
            image=img_tensors,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )

        for gen in generated_ids:
            decoded = tokenizer.decode(gen, skip_special_tokens=True)
            results.append(decoded)

    return results

def load_cells_for_page(page_hash):
    file_path = cells_dir / f"{page_hash}.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    return float(prediction.strip().lower() == ground_truth.strip().lower())

def track_run(run_func, images, prompts, batch_size=4):
    results = []
    batch_stats = []

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        reset_gpu_memory()
        start_ram = measure_ram()
        start_time = time.time()

        batch_results = run_func(batch_imgs, batch_prompts, batch_size=batch_size)

        latency = time.time() - start_time
        gpu_mem = measure_gpu_memory()
        ram_used = measure_ram() - start_ram
        batch_stats.append({
            "latency": latency,
            "gpu_mem": gpu_mem,
            "ram": ram_used,
            "batch_size": len(batch_imgs)
        })

        results.extend(batch_results)
        print(f"finished batch {i}")

    return results, batch_stats

def benchmark_model(model_name, run_func, dataset, batch_size=4):
    vqa_predictions = []
    vqa_references = []

    images = [sample["image"] for sample in dataset]
    ocr_prompts = ["Extract all text from this document." for _ in dataset]
    vqa_prompts = [sample["question"] for sample in dataset]

    ocr_predictions, ocr_stats = track_run(run_func, images, ocr_prompts, batch_size)
    ocr_references = [sample["ocr"] for sample in dataset]

    if isinstance(ocr_predictions[0], dict) and "answer" in ocr_predictions[0]:
        ocr_predictions = [p["answer"] for p in ocr_predictions]

    vqa_predictions, vqa_stats = track_run(run_func, images, vqa_prompts, batch_size)
    vqa_references = [sample["answer"] for sample in dataset]
    
    if isinstance(vqa_predictions[0], dict) and "answer" in vqa_predictions[0]:
        vqa_predictions = [p["answer"] for p in vqa_predictions]

    cer_score = cer_metric.compute(predictions=ocr_predictions, references=ocr_references)
    wer_score = wer_metric.compute(predictions=ocr_predictions, references=ocr_references)

    vqa_accuracy_score = np.mean([vqa_accuracy(p, r) for p, r in zip(vqa_predictions, vqa_references)])
    
    return {
        "ocr": {
            "cer": cer_score,
            "wer": wer_score,
            "batch_stats": ocr_stats
        },
        "vqa": {
            "accuracy": vqa_accuracy_score,
            "batch_stats": vqa_stats
        }
    }



print("VLM BENCHMARKING SCRIPT")
print(f"Running on device {DEVICE}")

print("Setup for models")
(smol_model, smol_processor) = smolVLM_setup()
moon_model = moondream_setup()
nano_model = nanoVLM_setup()

print("Setup for benchmarks")
# using DocLayNet as dataset
# tasks related to: visual question answering, optical character recognition
# collect accuracy metrics, measure latency and amount of memory used for each task

# raw_data = load_dataset("ds4sd/DocLayNet", split="test", download_config=DownloadConfig(num_proc=1, max_retries=20))
#raw_data = load_dataset("coco", data_dir="../../DocLayNet/COCO", split="train")

with open("../../DocLayNet/COCO/train.json") as f:
    annotations = json.load(f)
    annotations = annotations
    annotations["images"] = annotations["images"][:300]

print("JSON opened")
length = len(annotations["images"])

ann_index = defaultdict(list)
for a in annotations["annotations"]:
    ann_index[a["image_id"]].append(a)

print('indexed')
samples = []
last_check = 0
for i, ann in enumerate(annotations["images"]):
    img_path = "../../DocLayNet/PNG/" + ann["file_name"]
    ocr = ann_index[ann["id"]]
    samples.append({
        "img_path": img_path,
        "ocr": ocr,
        "page_hash": ann["file_name"].split(".")[0]
    })
    # progress update print
    completion = 100 * i / length
    if completion - last_check >= 10:
        print(f"Completion: {100 * i / length}%")
        last_check = completion


print("samples built")
def build_benchmark(sample):
    image = Image.open(sample["img_path"]).convert("RGB")
    image = resize_for_smol(image)

    cells_data = load_cells_for_page(sample["page_hash"])
    cells = cells_data["cells"]

    full_text = []
    for ann in sample["ocr"]:
        if ann["category_id"] == 6:  # text block
            x, y, w, h = ann["bbox"]
            for cell in cells:
                cx, cy, cw, ch = cell["bbox"]
                if (x < cx + cw and x + w > cx and y < cy + ch and y + h > cy):
                    text = cell["text"].strip()
                    if text:
                        full_text.append(text)
    
    if len(full_text) == 0:
        ocr_text = ""
    else:
        ocr_text = "\n".join(full_text)

    question = "How many text blocks are present in this document?"
    answer = str(len([ann for ann in sample["ocr"] if ann["category_id"] == 6]))

    return {
        "image": image,
        "ocr": ocr_text,
        "question": question,
        "answer": answer
    }

dataset = Dataset.from_list(samples)
benchmark_dataset = dataset.map(build_benchmark, num_proc=4)
print("Benchmarking models")


print("TESTING NANO:")
nano_results = benchmark_model(
    "nanoVLM",
    lambda imgs, prompts, batch_size: nanoVLM_run(nano_model, imgs, prompts, batch_size=batch_size),
    benchmark_dataset
)

print("TESTING SMOLVLM:")
smol_results = benchmark_model(
    "smolVLM",
    lambda imgs, prompts, batch_size: smolVLM_run(smol_model, smol_processor, imgs, prompts, batch_size=batch_size),
    benchmark_dataset
)

print("TESTING MOONDREAM:")
moondream_results = benchmark_model(
    "moondream",
    lambda imgs, prompts, batch_size: moondream_run(moon_model, imgs, prompts, batch_size=batch_size),
    benchmark_dataset
)



print("TKTK: Generating output (NOT IMPLEMENTED)")

# TODO this is a placeholder function
# we should probably be writing data to a CSV or something instead of this so we can use the data elsewhere

def summarize(name, results):
    print(f"=== {name} ===")
    for task in ["ocr", "vqa"]:
        stats = results[task]["batch_stats"]
        avg_latency = np.mean([s["latency"] for s in stats])
        avg_gpu = np.mean([s["gpu_mem"] for s in stats])
        avg_ram = np.mean([s["ram"] for s in stats])
        print(f"[{task.upper()}] avg_latency: {avg_latency:.4f}s, avg_gpu: {avg_gpu:.2f}MB, avg_ram: {avg_ram:.2f}MB")
    print(f"CER: {results["ocr"]["cer"]} WER: {results["ocr"]["wer"]} VQA_accuracy: {results["vqa"]["accuracy"]}")

summarize("SmolVLM-256m", smol_results)
summarize("Moondream-0.5b", moondream_results)
summarize("NanoVLM", nano_results)


print("DONE!")