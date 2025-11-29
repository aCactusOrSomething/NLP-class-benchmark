okay

so

still needs a bunch of cleanup. the isntructions for setting up nanoVLM are here:

https://github.com/huggingface/nanoVLM/tree/v0.1

requirements.txt isn't complete. you can install the latest versions of whatever packages that python says you need, EXCEPT for moondream, which must be 

pip install moondream=0.0.5

the moondream model file is still way too big to include in the repository.

you can download it here:

https://huggingface.co/vikhyatk/moondream2/resolve/onnx/moondream-0_5b-int8.mf.gz?download=true

but it will need to be unzipped.