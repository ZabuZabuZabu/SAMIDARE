# SAMIDARE: Advanced Tracking-by-Segmentation for dence environment

## 📝 Abstract
Automated sports analysis demands robust multi-object tracking (MOT), yet segmentation-based methods often struggle with mask errors and ID switches in dense scenes. We propose SAMIDARE, a framework that enhances SAM2MOT for congested scenarios through two key components: (1) adaptive mask control, integrating density-aware mask re-generation and selective memory updates to preserve feature integrity; and (2) state-aware association and new track initialization, which improves robustness under occlusions and frequent frame-out events. Evaluated on SportsMOT, SAMIDARE achieves state-of-the-art performance, outperforming the baseline by 2.5 HOTA and 4.2 IDF1 points on the validation set. These results demonstrate that adaptive feature management and state-aware association provide a robust and efficient solution for dense sports tracking.

## 🛠 Setup & Installation

### 1. Clone SAM2 Repository
This project requires the SAM2 environment. Follow the official instructions(https://github.com/facebookresearch/sam2?tab=readme-ov-file) to clone and install the base model:

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
