# Binary Image Classifier — Cats vs Dogs

A convolutional neural network (CNN) trained from scratch to classify images into two categories.

## Overview

This project implements a binary image classifier using PyTorch. The model is trained on a dataset of 8000 images (4000 per class) and evaluated on a held-out test set.

## Dataset

- **Classes:** Cats, Dogs
- **Training set:** 4000 images/class
- **Source:** [Dogs vs Cats — Kaggle]([https://www.kaggle.com/datasets/chetankv/dogs-cats-images])

## Model Architecture

Custom CNN built with PyTorch:

| Layer       | Details                        |
|-------------|-------------------------------|
| Conv1       | 3 → 6 filters, kernel 5x5     |
| MaxPool     | 2x2                           |
| Conv2       | 6 → 16 filters, kernel 5x5    |
| MaxPool     | 2x2                           |
| FC1         | auto → 120                    |
| FC2         | 120 → 84                      |
| FC3         | 84 → 2 (output)               |

- **Input size:** 224 × 224 × 3 (RGB)
- **Loss function:** CrossEntropyLoss
- **Optimizer:** SGD (lr=0.01, momentum=0.9)
- **Epochs:** 5

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~75%   |

## Project Structure
```
├── train/
│   ├── cats/
│   └── dogs/
├── test/
│   ├── cats/
│   └── dogs/
├── model.pth
└── notebook.ipynb
```

## Requirements
```
torch
torchvision
Pillow
```

## Usage

**Train:**
```python
model = Net()
# run training cells in notebook.ipynb
```

**Inference on a single image:**
```python
img = Image.open("image.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)

model.eval()
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    print(classes[predicted.item()])
```

## Notes

- All images are resized to 224×224 at load time to ensure consistent tensor shapes
- `.convert("RGB")` is applied at inference to handle grayscale or RGBA inputs
- The flatten size after conv layers is computed automatically via a dummy forward pass
