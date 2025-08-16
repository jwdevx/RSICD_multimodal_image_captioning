# README — Project Notebook Navigation Guide

This document provides a **navigation guide** for the Jupyter Notebook **"Code_Geass_final_merged.ipynb"** used in this project and is  located at **/Notebook folder**.  

## Folder Contents

The **`/Notebook`** folder contains:

- **`Code_Geass_final_merged.ipynb`**  
- Images used within the notebook to explain architectures and visualize results:  
  - `BLIP.jpg`  
  - `file-structure.jpg`  
  - `lam.jpg`  
  - `Model-performance.jpg`  
  - `multi-object-scenes.jpg`  
  - `Plateau5.jpg`  
  - `VitGpt2.jpg`  

## Important

These images must be stored in the same folder as the Jupyter Notebook to ensure all visual references display correctly when viewing the notebook.



The notebook contains problem statement and significance, data analysis , implementation of multiple models and analyses of the  results. These sections are  organized with headings and subheadings along with table of content for the entire notebook and individual sections,  so that relevant sections can be nevigated to quickly.

---

## **1. Introduction**
- **Motivation** – Overview of why the project was undertaken.
- **Problem Statement** – The main challenges addressed.

---
## **2.Data Sources – RSICD**
 – Description of dataset.
 --reference

---

## **3. Exploratory Analysis**
- **Challenges (1–4)** – Scale Variation, Semantic Ambiguity, Rotation Ambiguity, Multi-Object Scenes.
- **Per Class Analysis** – Class-wise dataset breakdown.
- **Caption Analysis** – Overall Caption analysis.
- **Individual Caption Analysis** – Closer inspection of sample captions.
- **Preprocessing Pipeline** – Steps applied before training.

---

## **4. Model Sections**

Each model follows a similar structure:
1. **Architecture** – Detailed description of the model’s design.
2. **Model Hyperparameters** – Configuration used for training.
3. **Table of Content** – Quick navigation to results and analyses for the model.
4. **Results** – Performance metrics and plots.
5. **Generated Captions** – Sample results by the trained model.
6. **Conclusion / Key Observations** – Summary of performance and limitations.

---

### **Method 1 – LAM **
- **LAM Architecture**
- **LAM Model Hyperparameters**
- **LAM Table of Content**
- **LAM Results**
- **Generated Captions by LAM**
- **LAM Conclusion**

---

### **Method 2 – ViT + GPT-2**
- **ViT-GPT2 Architecture**
- **ViT-GPT2 Model Hyperparameters**
- **ViT-GPT2 Table of Content**
- **ViT-GPT2 Results**
- **Generated Captions by ViT-GPT2**
- **ViT-GPT2 Conclusion**

---

### **Method 3 – BLIP (Bootstrapping Language-Image Pretraining)**
#### Data Preprocessing  
- Conversion of RSICD JSON → COCO format.

---

#### **Blip1 Base**
- **Blip1 (Bootstrapped Language-Image Pretraining)**
- **Architecture**
- **Finetuning on the RSICD Dataset**
- **Table of Content**
- **Plots (Loss & Accuracy)**
- **Results**
- **Data Augmentation** – Strategy and transformations.
- **Augmentation Results**
- **Generated Image Captions**
- **Model Selection** – Final model selection for finetuning.

---

#### **Blip1 Large**
- **Specifications**
- **Motivation for Larger Model**
- **Training Specifications**
- **Table of Content**
- **Plots (Loss & Accuracy)**
- **Evaluation on Test Set**
- **Comparison with Blip1 Base**
- **Caption Comparison on Specific Images**

---

## **4. Discussion**
- **Blip1 vs ViT-GPT2 vs LAM** – Performance trade-offs.
- **Blip1 Large vs Blip1 Base** – Gains from scaling up.
- **Key Observations** – Major takeaways.
- **External Models** – Comparisons with already existing models, namely VLAD+RNN and SkyEyeGPT.

---

## **5. Conclusion**
- Final remarks, summarizing which models performed best and why + challenges encountered.

---

### **Navigation Tips for each section**
- Use the **Table of Contents (ToC)** on the left panel in Jupyter for quick access.
- For **results and metrics**, navigate directly to the “Results” or “Comparison” subheadings under each model.
- **Generated captions** are clearly labeled for qualitative analysis.
- **Key Observations** and **Conclusion** sections provide a condensed view of findings without reading through the entire notebook.

---
