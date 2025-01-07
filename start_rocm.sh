#!/bin/bash
export PYTORCH_ROCM_ARCH="gfx1031"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm
./.venv/bin/python main.py