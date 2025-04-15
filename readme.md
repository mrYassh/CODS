# Camouflage Segmentation Project

This project uses a conda environment for dependency management. Follow the steps below to set up and run the project.

## Prerequisites

- Anaconda or Miniconda installed on your system
- Python 3.8 or higher

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd camouflage-segmentation
```

2. Create and activate the conda environment using the provided environment.yml file:
```bash
conda env create -f environment.yml
conda activate camouflage
```

## Running the Application

1. Make sure you're in the project directory and the conda environment is activated:
```bash
conda activate camouflage
```

2. Run the application:
```bash
python app.py
```

## Environment Details

The project uses the following key dependencies:
- Python 3.8.20
- PyTorch 2.4.1
- OpenCV 4.11.0.86
- Flask 2.0.1
- Other dependencies as specified in environment.yml

## Walkthrough Video

Watch our step-by-step walkthrough video to learn how to set up and use the project:

[![Camouflage Segmentation Walkthrough](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

The video covers:
- Project overview and features
- Environment setup
- Running the application
- Using the interface
- Common use cases and examples

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are properly installed by running `conda list`
2. Try recreating the environment if you face dependency conflicts
3. Make sure you're using the correct Python version (3.8.20)
