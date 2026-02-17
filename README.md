# SAMIDARE: Advanced Tracking-by-Segmentation method for dence environment

## 📝 Abstract
Automated sports analysis relies heavily on robust multi-object tracking (MOT) to quantify player performance and decode complex tactical patterns. Nevertheless, the sports domain presents formidable obstacles, including erratic athlete trajectories, persistent interpersonal occlusions, and dynamic background changes caused by rapid pan-tilt-zoom camera operations. While traditional tracking-by-detection methods remain vulnerable to detector noise, existing segmentation-based approaches often struggle to maintain stable trajectories in overcrowded environments. To address these limitations, we propose SAMIDARE, a tracking-by-segmentation framework that enhances the baseline SAM2MOT for high-density sports scenarios. SAMIDARE integrates density-aware module to regulate mask refinement and employs an adaptive memory update control module to maintain identity consistency during complex player interactions. Evaluated on the SportsMOT dataset, SAMIDARE achieves state-of-the-art performance, demonstrating superior accuracy in handling dense environments. Our results highlight the benefits of density-aware mask propagation and robust memory management for a more generalizable MOT approach in the demanding domain of sports tracking.

## 🛠 Setup & Installation

### 1. Clone SAM2 Repository
This project requires the SAM2 environment. Follow the official instructions to clone and install the base model:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### 2. Download SAM2 Checkpoints

Download the SAM2 checkpoints (e.g., sam2.1_hiera_large.pt) from the official SAM2 repository and place them in the checkpoints/ directory.

### 3. Add SAMIDARE Scripts
Copy the following three files from this repository into the SAM2 root directory:
```bash
samidare_for_vis.py (For visualization)
samidare_for_test.py (For text output and testing)
experiment.sh (For automated testing)
```

## 🏃 Execution
### 1. Data Preparation
Download the dataset and detection results (e.g., SportsMOT) from the provided Google Drive link.

Place the data in your workspace.

### 2. Configuration
Adjust the paths in the scripts to match your local environment:

```bash
--video_path: Path to the video sequence (containing img1/ and gt/gt.txt).
--output_dir: Directory to save tracking results and videos.
```

### 3. Running the Scripts
To run individual tracking with visualization:

```bash
python samidare_for_vis.py --video_path /path/to/data --output_video result.mp4
```

To run the automated test suite:
```bash
bash experiment.sh
```
