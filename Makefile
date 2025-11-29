# Current used Python version (LTS)
PYTHON_VERSION := 3.11

# Using CPU-Only version for now, so no CUDA dependencies
# Because of my limited hardware capabilities at the moment
# If you have a powerful GPU and want to use CUDA, uncomment the CUDA source line below
PYTORCH_SOURCE := https://download.pytorch.org/whl/cpu 		# Only support CPU ver (no CUDA)
# PYTORCH_SOURCE := https://download.pytorch.org/whl/cuda 	# Support CUDA ver (with CUDA)

# Additional pip install flags for PyTorch source
PIP_INSTALL_FLAGS := --extra-index-url $(PYTORCH_SOURCE) --index-strategy unsafe-best-match

.PHONY: init


# This installation required you to have UV installed
# If you don't have UV, please install it first from https://docs.astral.sh/uv/
# Then follow my project setup at SETUP.md file
init: uv pip install -r requirements.txt $(PIP_INSTALL_FLAGS)
