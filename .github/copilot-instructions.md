# SAM3 AI Agent Guide

## System Map
- `sam3/model_builder.py` is the single entry point for constructing detectors, trackers, and predictors; it stitches ViT vision backbones, VE text encoders, geometry encoders, DETR-style decoders, and the presence-token dot-product scorer.
- `sam3/model/` hosts tightly-coupled modules (e.g., `decoder.py`, `sam3_video_predictor.py`, `sam3_tracker_base.py`) while `sam3/agent/` wraps them for LLM-driven prompting and visualization hooks (`agent_core.py`, `viz.py`).
- Evaluation utilities live in `sam3/eval/` (cgF1, HOTA, COCO wrappers) and task-specific scripts under `scripts/eval/*` simply orchestrate configs plus result aggregation.
- Training relies on Hydra configs in `sam3/train/configs/`; launcher logic (`trainer.py`, `optim/`, `loss/`) mirrors these configs so editing YAML is preferred over ad-hoc CLI flags.

## Environment & Tooling
- Target Python 3.12 + CUDA 12.6 (see `.devcontainer/Dockerfile`); install PyTorch via `pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126` before extras.
- Always `pip install -e .` and add feature extras explicitly (`.[dev]`, `.[train]`, `.[notebooks]`) to pull formatter/test deps, Hydra+submitit, and notebook toolchains.
- Large checkpoints require an authenticated Hugging Face session (`hf auth login`) prior to running any builder helpers.

## Core Workflows
- **Image inference:** `build_sam3_image_model()` + `Sam3Processor` (`sam3/model_builder.py`, `sam3/model/sam3_image_processor.py`); set image via `processor.set_image`, then call `set_text_prompt` for masks/boxes.
- **Video inference:** `build_sam3_video_predictor()` returns a stateful predictor driven by `handle_request` messages (`sam3/model/sam3_video_predictor.py`).
- **Agent demos:** `examples/sam3_agent.ipynb` pairs `agent/client_llm.py` with `client_sam3.py` for multi-turn segmentation plans; reuse their helper functions when automating LLM loops.
- **Training / finetune:** `python sam3/train/train.py -c configs/<suite>/<job>.yaml --use-cluster {0,1} --num-gpus N`; configs control dataset roots, launcher, logging, so keep overrides minimal.
- **Eval benchmarks:** edit `sam3/train/configs/eval_base.yaml` paths, then run the provided commands in `scripts/eval/{gold,silver,veval}/README.md` (same `train.py` entry point) to dump predictions and cgF1 scores.

## Conventions & Quality
- Formatting: `ufmt format .` (wrapped `ruff`), import sorting via `usort`/`isort`; lint or type-check using `ruff`, `flake8`, `mypy` (configured for Python 3.12) before sending PRs.
- Tests: repo-level `pytest` targets `tests/` plus perf kernels under `sam3/perflib/tests`; keep CUDA-heavy tests behind `PYTEST_ADDOPTS="-m slow"` filters when possible.
- Logging favors `sam3/logger.py` helpers that already integrate rank-aware printing; prefer them over bare `print` in distributed flows.

## Data & Assets
- SA-Co datasets (gold/silver/veval) are referenced via Hugging Face or Roboflow; evaluation configs expect you to set absolute `paths.*` entries.
- For Roboflow/ODinW sweeps, job arrays (`submitit.job_array.*`) control which dataset split runs—adjust only the YAML index to keep automation consistent.

Let me know which sections need more depth or if other workflows should be documented.
