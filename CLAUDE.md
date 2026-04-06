# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FaceFusion is a face manipulation platform (v3.6.0) built in Python 3.10+. It provides face swapping, enhancement, editing, and other facial processing via a plugin-based processor system. It runs as a Gradio web UI or headless CLI.

## Commands

```bash
# Run
python facefusion.py run                  # Launch Gradio web UI
python facefusion.py headless-run         # CLI processing
python facefusion.py batch-run            # Batch processing with patterns

# Test
pytest tests/                             # Full test suite
pytest tests/test_face_enhancer.py        # Single test file
pytest tests/ --cov facefusion            # With coverage

# Lint & Type Check
flake8 facefusion.py install.py facefusion tests
mypy facefusion.py install.py facefusion tests

# Install
python install.py --onnxruntime default --skip-conda
```

## Architecture

### Entry Point & Flow
`facefusion.py` → `conda.setup()` → `core.cli()` → routes to command handler (run/headless-run/batch-run/job-*/benchmark). Processing flows through `facefusion/workflows/image_to_image.py` or `image_to_video.py`.

### Plugin Processor System
Processors in `facefusion/processors/modules/` are dynamically loaded via `importlib` in `facefusion/processors/core.py`. Each processor must implement these methods: `get_inference_pool()`, `clear_inference_pool()`, `register_args()`, `apply_args()`, `pre_check()`, `pre_process()`, `post_process()`, `process_frame()`.

Available processors: face_swapper, face_enhancer, face_editor, age_modifier, background_remover, deep_swapper, expression_restorer, frame_colorizer, frame_enhancer, lip_syncer, face_debugger.

### State Management
`facefusion/state_manager.py` holds a global state dict (`STATE_SET`) with CLI/UI contexts. Configuration loads from `facefusion.ini` (INI format) via `facefusion/config.py`. All args flow through `register_args()` → `apply_args()` → `state_manager.init_item()`.

### Face Analysis Pipeline
Detection → Landmarks → Classification → Recognition, orchestrated by `face_analyser.py`:
- `face_detector.py` – bounding boxes (retinaface, scrfd, yolo_face, yunet)
- `face_landmarker.py` – 5 or 68 facial landmarks
- `face_masker.py` – masks (box, occlusion, area, region)
- `face_selector.py` – filters by order, age, gender, race
- `face_classifier.py` – gender/age/race classification
- `face_recognizer.py` – embedding generation

### Inference
`inference_manager.py` pools ONNX InferenceSessions per processor. Models are defined via `create_static_model_set()` with hash-based download validation. Execution providers: CUDA, TensorRT, DirectML, ROCm, OpenVINO, CoreML, QNN, CPU.

### Video Processing Pipeline
Extract frames (FFmpeg) → process frames in parallel (ThreadPoolExecutor) → merge frames (FFmpeg) → restore audio. Frame processing applies each active processor in sequence per frame.

### Job System
`facefusion/jobs/` – JSON-based persistence in `.jobs/` directory. States: drafted → queued → running → completed/failed.

### UI
`facefusion/uis/` – Gradio components with layout definitions and component modules.

## Code Style

- **Indentation**: Tabs, not spaces (4-width)
- **Line endings**: LF
- **Imports**: pycharm-style ordering (stdlib → third-party → facefusion), enforced by flake8-import-order
- **List literals**: Spaces inside brackets `[ x, y ]`
- **Types**: Strict mypy (`disallow_untyped_defs`, `disallow_any_generics`, `disallow_untyped_calls`). Uses TypeAlias, Literal, TypedDict extensively.
- **Logging**: `logger.info(message, __name__)` – module name auto-formatted as `[FACEFUSION.MODULE]`

## Testing

Tests use pytest with module-scoped fixtures that conditionally download test assets from `facefusion-assets` GitHub releases. Most processor tests run `facefusion.py headless-run` as a subprocess and assert returncode == 0. Test helpers are in `tests/helper.py`.
