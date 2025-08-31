# 🌟AI-Driven Bone Tumor Classification and Segmentation🌟

This repository contains code, notebooks, and results for bone tumor classification and segmentation in X-rays using AI, transfer learning, and attention-based deep networks. The project leverages the BTXRD dataset and pre-trained models for improved accuracy and generalization.

---

## 1. Dataset Overview

- **BTXRD Dataset**: 3,746 labeled X-ray images with metadata (age, gender, body part, view), and 1,876 annotations (bounding boxes & polygonal tumor masks).
- **Class Distribution**: 
  - Normal: ~50.1%
  - Benign: ~40.7%
  - Malignant: ~10%
- **Age Distribution**: Right-skewed, with a peak around 10 years (pediatric cases dominate).
- **Gender Split**: 56% male  
  - Normal: 27%, Benign: 24%, Malignant: 5%

---

## 2. Preprocessing & Motivation

- **Human Vision Inspired Pipeline**:
  - First capture the image as a whole.
  - Then zoom into finer details using bounding box crops.

- **Normal Images**: Resized directly to 224×224.

- **Tumor Images**: Cropped using bounding boxes. Symmetric zero-padding used if crop < 224×224.

- **Augmentation**: Rotations, flips, and intensity scaling.

- **Class Imbalance Handling**: Weighted loss applied to improve malignant detection.

---

## 3. Tumor Classification Model

- **Base Model**: DenseNet121 (pre-trained on CheXpert)

- **Classifier Head** (non-LaTeX form):

**Dropout(ReLU(BatchNorm(Linear(512)))) → Linear(256) → Softmax**


- **Two-Phase Training**:
- **Phase 1**: 
  - Backbone frozen
  - Trained only the classifier head
  - Optimizer: Adam
  - LR scheduler (factor = 0.3), early stopping
- **Phase 2**:
  - Unfroze DenseBlock4, fine-tuned entire network
  - Optimizer: AdamW
  - Techniques: Gradient clipping (L2 norm ≤ 1), L2 regularization

- **Loss Function**: Weighted Cross-Entropy  
L_cls = - Σ (w_i * y_i * log(ŷ_i)) for i in classes


---

## 4. Segmentation Model

- **Architecture**: Improved U-Net with Attention Gates

- **Attention Gate Formulation**:

ψ = Sigmoid(Conv(ReLU(Conv(F_g) + Conv(F_l))))

- F_g: Gating signal
- F_l: Skip connection features

- **Loss Function**: Compound Loss
L_seg = α * BCE + (1 - α) * (1 - Dice)


- **Dice Score**:
Dice = 2 * |P ∩ G| / (|P| + |G|)


---

## 5. Results

### 🔍 Classification Performance

| Model | Train | Validation | Test |
|-------|-------|------------|------|
| DenseNet121 (CheXpert, Frozen) | 0.85 | 0.87 | 0.87 |
| DenseNet121 (CheXpert, Fine-Tuned) | 0.90 | 0.85 | 0.87 |

### 🌟 Multi-Class Metrics (Test Set)

| Class     | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
|-----------|-----------|--------|----------|---------|--------|
| Normal    | 1.00      | 0.96   | 0.98     | 1.00    | 1.00   |
| Benign    | 0.87      | 0.86   | 0.86     | 0.96    | 0.93   |
| Malignant | 0.44      | 0.53   | 0.48     | 0.87    | 0.42   |

- **Overall Accuracy**: **87%**

- **Binary Classification (Normal vs Tumor)**:
- Accuracy: **97%**
- Precision/Recall: ~0.96–1.00

- **Note**: Malignant detection remains challenging due to class imbalance.

### 🌟 Segmentation Results

| Metric | Basic U-Net | Improved U-Net |
|--------|-------------|----------------|
| Dice   | ~0.000      | 0.529          |
| IoU    | 0.505       | 0.375          |
| ROC AUC| –           | 0.949          |

---

## 6. Key Observations & Future Work

- Bounding box preprocessing significantly enhanced classification accuracy.
- Gradient clipping during fine-tuning helped reduce overfitting.
- **Malignant detection still lags** — potential future directions:
- Vision Transformers
- Semi-supervised learning
- Custom compound loss functions
- Full Grad-CAM integration (limited by hardware)
- Curriculum learning on A100 GPUs

---

## 🌟 References

- Yao, S., Huang, Y., Wang, X. et al.  
*A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors*.  
Sci Data 12, 88 (2025). [https://doi.org/10.1038/s41597-024-04311-y](https://doi.org/10.1038/s41597-024-04311-y)

- Huang, G., Liu, Z., van der Maaten, L., Weinberger, K.  
*Densely Connected Convolutional Networks*.  
CVPR 2017. [https://ieeexplore.ieee.org/document/8099726](https://ieeexplore.ieee.org/document/8099726)

