# ROBOP installation targets. Run `make help` for the complete list.
# Fresh environment: make setup assets models check
# Existing torch/PyTorch3D environment: pip install -r requirements.txt && make install
# The bundled PyTorch3D wheel matches the default PY/CUDA/TORCH values; custom
# combinations require a matching wheel or source build.

# PY is the wheel's Python tag; PYTHON is the interpreter.
CUDA   ?= cu121
TORCH  ?= 241
PY     ?= 310
PYTHON ?= python

TORCH_VERSION      = 2.4.1
TORCHVISION_VERSION = 0.19.1
TORCHAUDIO_VERSION  = 2.4.1

TORCH_WHEEL_BASE = https://download.pytorch.org/whl/$(CUDA)

NEMO_CKPT_DIR = external/NeMO/checkpoints
NEMO_CKPT     = $(NEMO_CKPT_DIR)/checkpoint.pth

GIGAPOSE_CKPT_DIR = external/gigapose/pretrained
GIGAPOSE_CKPT     = $(GIGAPOSE_CKPT_DIR)/gigaPose_v1.ckpt
GIGAPOSE_CKPT_URL = https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/gigaPose_v1.ckpt

# MegaPose refiner checkpoints (for estimator.refine=true). Layout + URLs mirror
# external/gigapose/src/scripts/download_megapose.py.
MEGAPOSE_DIR = external/gigapose/pretrained/megapose-models
MEGAPOSE_URL = https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models

# robot-renderer asset generation. owi535 and meca500 assets carry no usable
# redistribution license, so robot-renderer ships a script that downloads the
# originals (SHA-256 pinned) and re-applies its modifications locally.
RR_DIR        = external/robot-renderer
RR_ASSET_TOOL = $(RR_DIR)/tools/fetch_assets.py
RR_DOWNLOADS  = $(RR_DIR)/tools/_downloads

.PHONY: help setup install-torch install-pt3d install install-dev submodules \
        assemble-checkpoint remove-nemo-checkpoint-shards download-gigapose \
        download-megapose install-gigapose \
        models assets clean-asset-cache check

# Target index
help:
	@echo "ROBOP targets:"
	@echo ""
	@echo "  Install (in order, from a bare conda env):"
	@echo "    make setup            torch -> pytorch3d -> requirements.txt -> local packages"
	@echo "    make assets           owi535 + meca500 meshes robot-renderer cannot ship"
	@echo "    make models           all model weights + GigaPose's env deps"
	@echo "    make check            verify the key imports"
	@echo ""
	@echo "  Pieces of 'make setup':"
	@echo "    make install-torch    PyTorch CUDA wheels (2.4.1 + cu121 by default)"
	@echo "    make install-pt3d     fvcore, iopath, and the pinned PyTorch3D wheel"
	@echo "    make install          local editable packages (NeMO, robot-renderer, robop)"
	@echo "    make submodules       git submodule update --init --recursive"
	@echo ""
	@echo "  Pieces of 'make models':"
	@echo "    make assemble-checkpoint  concatenate the NeMO checkpoint shards"
	@echo "    make install-gigapose     bop_toolkit, panda3d, pinocchio (needs conda)"
	@echo "    make download-gigapose    gigaPose_v1.ckpt"
	@echo "    make download-megapose    MegaPose coarse-rgb + refiner-rgb checkpoints"
	@echo ""
	@echo "  Other:"
	@echo "    make install-dev          make install + pytest"
	@echo "    make clean-asset-cache    delete the asset download cache"
	@echo "    make remove-nemo-checkpoint-shards  save disk; dirties the NeMO submodule"
	@echo ""
	@echo "Every target is idempotent - re-running skips what is already present."

# Full setup from a bare conda env
setup: install-torch install-pt3d
	pip install -r requirements.txt
	$(MAKE) install

# Step 1: PyTorch CUDA wheels
install-torch:
	pip install \
		$(TORCH_WHEEL_BASE)/torch-$(TORCH_VERSION)%2B$(CUDA)-cp$(PY)-cp$(PY)-linux_x86_64.whl \
		$(TORCH_WHEEL_BASE)/torchvision-$(TORCHVISION_VERSION)%2B$(CUDA)-cp$(PY)-cp$(PY)-linux_x86_64.whl \
		$(TORCH_WHEEL_BASE)/torchaudio-$(TORCHAUDIO_VERSION)%2B$(CUDA)-cp$(PY)-cp$(PY)-linux_x86_64.whl

# Step 2: PyTorch3D prebuilt wheel (precompiled, no nvcc needed)
install-pt3d:
	pip install fvcore==0.1.5.post20221221 iopath==0.1.10
	pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl

# Step 3: local editable packages
install: submodules assemble-checkpoint
	pip install -e external/NeMO --no-deps
	pip install -e external/robot-renderer --no-deps
	pip install -e . --no-deps
	plotly_get_chrome || echo "WARNING: plotly_get_chrome not found - install kaleido or run it manually"

submodules:
	git submodule update --init --recursive

install-dev: install
	pip install pytest

# Step 4: non-redistributable robot assets (owi535, meca500)
# The asset tool downloads and rebuilds meshes that cannot be redistributed.
# Its two extra packages are required only for generation.
ROBOT ?= all
FORCE ?=
assets: submodules
	pip install meshoptimizer==0.2.30a0 pycollada==0.9.3
	$(PYTHON) $(RR_ASSET_TOOL) --robot $(ROBOT) $(FORCE)
	@echo "Removing the asset download cache..."
	@rm -rf $(RR_DOWNLOADS)
	@echo "Assets ready under $(RR_DIR)/src/robot_renderer/robots/"

clean-asset-cache:
	rm -rf $(RR_DOWNLOADS)

# Step 5: every model weight, plus GigaPose's non-PyPI env deps
# This includes install-gigapose, which needs conda on PATH (pinocchio).
# Run the individual targets instead if you only need one model.
models: assemble-checkpoint install-gigapose download-gigapose download-megapose
	@echo "All model weights are in place."

# Assemble checkpoint shards
assemble-checkpoint:
	@if [ -f "$(NEMO_CKPT)" ]; then \
		echo "$(NEMO_CKPT) already exists - skipping."; \
	elif ls $(NEMO_CKPT_DIR)/part_* 1>/dev/null 2>&1; then \
		echo "Concatenating checkpoint shards..."; \
		cat $(NEMO_CKPT_DIR)/part_* > $(NEMO_CKPT); \
		echo "Done: $(NEMO_CKPT)"; \
	else \
		echo "No part_* shards found in $(NEMO_CKPT_DIR) - nothing to do."; \
	fi

remove-nemo-checkpoint-shards:
	@echo "Removing tracked NeMO shards; external/NeMO will be dirty."
	rm -f $(NEMO_CKPT_DIR)/part_*
	@echo "Restore before evaluation: git -C external/NeMO restore -- checkpoints/part_*"

# GigaPose non-PyPI dependencies. bop_toolkit_lib (--no-deps avoids conflicting
# extras) + panda3d (template rendering via call_panda3d; prerelease index, per
# GigaPose's install_env.sh). All other GigaPose deps are in requirements.txt.
install-gigapose:
	pip install --no-deps git+https://github.com/thodan/bop_toolkit.git@cea62d651c7e395b2e1962b9749e4e89693c6ac4
	pip install --pre --extra-index-url https://archive.panda3d.org/ panda3d==1.11.0.dev3233
	# pinocchio - SE(3) Transform used by megapose.lib3d. Install via conda
	# (GigaPose's tested 2.6.20): the modern pip `pin` (4.x) forces numpy>=2,
	# which uninstalls ROBOP's pinned numpy<2 and breaks robot-renderer/scipy.
	conda install -y -c conda-forge pinocchio=2.6.20
	@echo "Next: make download-gigapose  (fetches gigaPose_v1.ckpt)"

# Download GigaPose checkpoint (coarse model)
# DINOv2 backbone is fetched automatically via torch.hub on first run; the
# MegaPose refiner weights are NOT needed for the coarse estimator.
download-gigapose:
	@if [ -f "$(GIGAPOSE_CKPT)" ]; then \
		echo "$(GIGAPOSE_CKPT) already exists - skipping."; \
	else \
		mkdir -p $(GIGAPOSE_CKPT_DIR); \
		echo "Downloading GigaPose checkpoint..."; \
		wget -O $(GIGAPOSE_CKPT) $(GIGAPOSE_CKPT_URL); \
		echo "Done: $(GIGAPOSE_CKPT)"; \
	fi

# Download MegaPose refiner checkpoints (for estimator.refine=true)
# coarse-rgb + refiner-rgb (checkpoint.pth.tar + config.yaml each), idempotent.
download-megapose:
	@for m in coarse-rgb-906902141 refiner-rgb-653307694; do \
		mkdir -p $(MEGAPOSE_DIR)/$$m; \
		for f in checkpoint.pth.tar config.yaml; do \
			if [ -f "$(MEGAPOSE_DIR)/$$m/$$f" ]; then \
				echo "$$m/$$f exists - skipping."; \
			else \
				wget -O $(MEGAPOSE_DIR)/$$m/$$f $(MEGAPOSE_URL)/$$m/$$f; \
			fi; \
		done; \
	done
	@echo "Set estimator.megapose_models_root=$(abspath $(MEGAPOSE_DIR))"

# Sanity check: verify key imports work
check:
	python -c "import torch; print('torch:', torch.__version__)"
	python -c "import pytorch3d; print('pytorch3d: ok')"
	python -c "import robot_renderer; print('robot_renderer: ok')"
	python -c "from nemolib.model import Model; print('nemolib: ok')"
	python -c "from robop import NeMORobotPoseEstimator; print('robop: ok')"
