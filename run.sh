uv init --bare --python 3.12
uv sync --python 3.12
source .venv/bin/activate
uv add torch numpy torchvision pillow datasets huggingface-hub transformers wandb
pip install torch numpy torchvision pillow datasets huggingface-hub transformers wandb