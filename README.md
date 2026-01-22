# Deep Learning Final Project: Mobile Gallery Image Classification

**Student:** Hasan Aksoy  
**Student Number:** 140130  
**Course:** SGH Deep Learning  

## Project Overview

This project implements a Convolutional Neural Network (CNN) for multi-class image classification, designed to automatically organize mobile phone gallery images into six distinct categories: Cars, Memes, Mountains, Selfies, Trees, and WhatsApp Screenshots.

The implementation utilizes Julia programming language with the Flux.jl deep learning framework, demonstrating the application of neural networks to a practical image classification task.

## Repository Contents

- `DL_Final_Project.ipynb` — Complete Jupyter Notebook with implementation and analysis
- `DL_Final_Project.html` — HTML export of the notebook for viewing
- `Project.toml` — Julia project dependencies
- `test_notebook.jl` — Standalone test script for verification

## Dataset

The dataset used in this project is the **Mobile Gallery Image Classification Dataset**, available on Kaggle:

📥 **Download Link:** [https://www.kaggle.com/datasets/n0obcoder/mobile-gallery-image-classification-data](https://www.kaggle.com/datasets/n0obcoder/mobile-gallery-image-classification-data)

### Dataset Setup Instructions

1. Download the dataset from the Kaggle link above
2. Extract the contents to the `data/` directory within this project
3. Ensure the following directory structure:

```
DL-FinalProject/
├── data/
│   └── train/
│       ├── Cars/
│       ├── Memes/
│       ├── Mountains/
│       ├── Selfies/
│       ├── Trees/
│       └── Whatsapp_Screenshots/
├── DL_Final_Project.ipynb
└── Project.toml
```

## Running the Project

1. Install Julia (version 1.9 or higher recommended)
2. Navigate to the project directory
3. Install dependencies and launch Jupyter:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using IJulia; notebook()'
```

4. Open `DL_Final_Project.ipynb` and execute all cells

## Requirements

- Julia 1.9+
- Flux.jl
- Images.jl
- Plots.jl

All dependencies are specified in `Project.toml` and will be installed automatically.

## License

This project is submitted as part of the SGH Deep Learning course final project requirements.
