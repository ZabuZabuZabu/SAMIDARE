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

Download the SAM2 checkpoints (e.g., sam2.1_hiera_large.pt) from the official SAM2 repository and place them in /sam2/checkpoints.

### 3. Add SAMIDARE Scripts
Copy the following files from this repository into the SAM2 root directory:
```bash
samidare_for_vis.py
```
### 3. Setup for Detector

Download the checkpoints (e.g., sam2.1_hiera_large.pt) from the following Google Drive (https://drive.google.com/drive/folders/1KRYiCUdtT0IA4YxhXC16S350W2FqR2og?usp=sharing)
and place it in /yolox/checkpoints.

Install the requirements.
```bash
pip install -r requirements.txt
```

## 🏃 Demo
### 1. Data Preparation
you can use our demo data which we have already put in our repository, but if you want to run with your own video, please run under code

```bash
python detection.py --video "path_to_your_mp4_file"
```

### 2. Running the Scripts
Before you run the scripts, please check if your directory structure is as follows:
```bash
ROOT
|--sam2
|   |--sam2
|   |   :
|   |--samidare_for_vis.py
|
|--detection
|   |--demo_data
|   |   |--det
|   |   |   |--det.txt
|   |   |
|   |   |--img1
|   |       |--000001.jpg
|   |             :
|   |             
|   |--your_own_data
|       |--det
|       |   |--det.txt
|       |
|       |--img1
|           |--000001.jpg
|                 :
|
|--yolox
|--detection.py
|--requirement.txt
|--your_own_video.mp4
```

Run under code. If you run with your own data, please change --data_path
```bash
cd sam2
python samidare_for_vis.py --output_dir "path_to_your_output_directory" --data_path /detection/demo_data
```
