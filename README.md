# simil.art | _Université Paris Cité_

## Overview

simil.art is an application for Content-Based Image Retrieval (CBIR) based on deep features extracted from neural networks, specifically ResNet18 and fine-tuned ResNet50 on an artistic dataset. The purpose of this project is to work with an artistic dataset, retrieve similar images, and refine the results based on both the style of the query image (using fine-tuned models) and the color (using color histogram correlation).

## Features

- Content-Based Image Retrieval (CBIR)
- Deep features extraction using ResNet18
- Fine-tuned model using ResNet50
- Refinement of results based on image style and color histogram correlation

## Getting Started

To get started with the simil.art project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yassinefkh/simil.art.git
   ```

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

1. Download the large files containing feature vectors and Faiss indexes using Git LFS:

   ```bash
   git lfs pull
   ```

## Usage

To run the application, execute the following commands:

```bash
 python manage.py
```

Then, open a web browser and navigate to [http://localhost:8000](http://localhost:8000).

Upload an image to search for similar images.

## Disclaimer

This project uses data from [WikiArt](https://www.wikiart.org/) for educational purposes only and is not intended for commercial use.

## Authors

- **Project Author:** FEKIH HASSEN Yassine, CALMANOVIC-PLESCOFF Auguste, HIBAOUI Imane, MARTIN--SAVORNIN Alain
- **Project Supervisor:** Pr. KURTZ Camille
