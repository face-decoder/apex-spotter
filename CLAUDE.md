# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Apex-spotter** detects apex frames (peak expression moments) in micro-expression videos through facial landmark tracking, optical flow computation, and signal processing. Built for research with CASME-2 and SAMM datasets.

### Core Functionality
- **Temporal apex detection**: Identifies peak intensity frames in micro-expressions
- **Phase segmentation**: Extracts onset-apex-offset boundaries
- **Multi-ROI analysis**: Supports face, eyes, eyebrows, mouth regions
- **GPU acceleration**: Automatic CUDA fallback for optical flow

## Build & Run Commands

```bash
# Install dependencies (requires UV: https://docs.astral.sh/uv)
uv venv
uv sync

# Alternative via Makefile
make init
```

### Primary Usage

Work primarily through Jupyter notebooks in `/notebooks/`:
- `apex-frame-casme.ipynb` - CASME-2 dataset processing
- `apex-frame-samm.ipynb` - SAMM dataset processing  
- `apex-frame-spotting.ipynb` - General apex spotting experiments

**Note**: No test suite or linting configuration exists. Development is notebook-driven.

## Architecture

### Processing Pipeline

```
Video Input
  ↓
Frame Pair Extraction
  ↓
Face Detection (MediaPipe 478 landmarks)
  ↓
Face Alignment (Affine transform to standard pose)
  ↓
ROI Extraction (Eyes, eyebrows, mouth, full face)
  ↓
Optical Flow Computation (TV-L1, GPU-accelerated)
  ↓
Magnitude Aggregation (Per-ROI mean flow)
  ↓
Signal Smoothing (Savitzky-Golay filter)
  ↓
Peak Detection (scipy.signal.find_peaks)
  ↓
Phase Boundary Extraction (Onset/Offset via threshold)
```

### Module Structure

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| `src/image/` | Grayscale conversion, validation | `to_grayscale()` |
| `src/video/` | Frame iteration with functional pattern | `Video` |
| `src/face/` | Landmark detection, alignment, ROI extraction | `FaceLandmark`, `FaceAligner` |
| `src/optical_flow/` | Dense flow computation (GPU/CPU) | `TVL1` |
| `src/apex/` | Apex/phase detection, signal processing | `ApexPhase` |

### Key Classes & Methods

#### **Video** (`src/video/`)
Functional frame pair iteration:
```python
Video(path).map(lambda curr, next: process_pair(curr, next))
```

#### **FaceLandmark** (`src/face/`)
- `detect(image)` → 478 MediaPipe facial landmarks
- `crop_roi(image, landmark_result, roi_points, target_size)` → Extracted ROI

Supported ROIs (via `FaceRoiPoints`):
- `LEFT_EYE_POINTS`, `RIGHT_EYE_POINTS`
- `LEFT_EYEBROW_POINTS`, `RIGHT_EYEBROW_POINTS`
- `MOUTH_POINTS`, `FACE_OVAL_POINTS`

#### **FaceAligner** (`src/face/`)
- `align(image, landmark_result)` → Affine-transformed face
- Uses eye-nose triangle as reference points

#### **TVL1** (`src/optical_flow/`)
- `compute(prev_gray, next_gray)` → Dense flow field (x, y components)
- Automatic GPU detection with CPU fallback

#### **ApexPhase** (`src/apex/`)
- `find_apex(signal)` → List of apex indices (scipy peak detection)
- `find_phase(signal, apex_indices)` → Dict of `{apex: {start, end}}`
- Adaptive Savitzky-Golay smoothing parameters

### GPU Acceleration Pattern

All GPU-dependent code follows this pattern:
```python
if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Use cv2.cuda_OpticalFlowDual_TVL1.create()
else:
    # Fallback to cv2.optflow.DualTVL1OpticalFlow_create()
```

For GPU setup instructions, see `SETUP_WITH_CUDA.md`.

## Important Locations

| Path | Contents |
|------|----------|
| `/.local/pre-trained/` | MediaPipe face landmarker model |
| `/src/face/tasks/face_landmarker.task` | Bundled MediaPipe model file |
| `/packages/cv2.so` | Custom OpenCV build with CUDA support |
| `/notebooks/` | Jupyter notebooks (primary development interface) |

## Code Conventions

- **Language**: Code comments are in Indonesian
- **Validation**: Extensive input validation with descriptive `ValueError` messages
- **Module exports**: All public APIs exposed via `__init__.py`
- **Path management**: `module.py` handles notebook import path synchronization

### Typical Function Pattern
```python
def process(image: np.ndarray, param: float = 0.5) -> dict:
    """
    Singkat deskripsi dalam Bahasa Indonesia.
    
    Args:
        image: Input grayscale/color image
        param: Processing parameter (default: 0.5)
        
    Returns:
        Dictionary containing results
        
    Raises:
        ValueError: If image is empty or invalid dimensions
    """
    if image is None or image.size == 0:
        raise ValueError("Image tidak boleh kosong")
    # Implementation...
```

## Dependencies

**Core Stack**:
- **MediaPipe** (>=0.10.0) - Face landmark detection (478 points)
- **OpenCV** (custom CUDA build) - Video I/O, optical flow
- **CuPy** (>=12.0.0) - GPU array operations
- **SciPy** (>=1.11.0) - Peak detection (`find_peaks`)
- **NumPy**, **Pandas**, **Matplotlib** - Numerical/visualization

**Requirements**: Python >=3.10

### Installation Notes
- GPU support requires CUDA Toolkit 12.x
- Custom OpenCV build in `/packages/` for CUDA optical flow
- See `pyproject.toml` for complete dependency list

## Troubleshooting

### GPU Not Detected
```bash
# Verify CUDA availability
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"

# If returns 0, check:
# 1. CUDA Toolkit installed (nvcc --version)
# 2. Custom cv2.so in packages/ is loaded
# 3. LD_LIBRARY_PATH includes CUDA libs
```

### MediaPipe Model Not Found
```bash
# Model should exist at:
ls src/face/tasks/face_landmarker.task

# Re-download if missing (see setup instructions)
```

### Memory Issues with Large Videos
- Use `Video.map()` for streaming processing (avoids loading full video)
- Process ROIs individually rather than full frames
- Reduce target ROI sizes in `FaceRoiSizes`

## Performance Notes

- **TV-L1 Optical Flow**: ~10ms/frame (GPU), ~150ms/frame (CPU) at 640x480
- **MediaPipe Landmarks**: ~15ms/frame (CPU)
- **Savitzky-Golay Filter**: Window size should be 5-15% of signal length
- **Peak Detection**: Tune `height`, `distance`, `prominence` parameters per dataset

## Research Context

This tool implements apex frame detection for micro-expression spotting research. Key papers:
- **CASME-2**: Yan et al. (2014) - Spontaneous micro-expression database
- **TV-L1 Optical Flow**: Zach et al. (2007) - Robust motion estimation
- **Phase Segmentation**: Li et al. (2013) - Onset-apex-offset detection

Typical workflow: Extract frames → Detect apex → Validate against ground truth CSV → Compute F1/precision/recall metrics.