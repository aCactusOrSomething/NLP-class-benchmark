#! /usr/bin/bash
echo "SETTING UP PYTHON ENVIRONMENT"
uv init --bare --python 3.12
uv sync --python 3.12
source .venv/bin/activate
uv add torch numpy torchvision pillow==10.4.0 datasets huggingface-hub transformers wandb moondream==0.0.5 psutil evaluate jiwer --frozen
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "DOWNLOADING MOONDREAM (NOT IMPLEMENTED)"

python models/nanoVLM-0.1/main.py