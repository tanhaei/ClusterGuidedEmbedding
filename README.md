# **Temporal-Multimodal Cluster-Guided Embedding for Patient Similarity Search: An Interpretable EHR Retrieval Framework**

This repository contains the official implementation of the **Cluster-Guided Embedding** framework for patient similarity search in Electronic Health Records (EHR), with a focus on ophthalmology cases from the BioArc registry.

## **Overview**

Our framework structures heterogeneous EHR data into clinically meaningful clusters (e.g., demographics, examinations, clinical notes). By learning cluster-specific embeddings, we reduce noise from irrelevant features and improve both retrieval accuracy and clinical interpretability.

### **Key Features**

* **Clinically Guided Clustering:** Groups features using domain expertise and unsupervised methods (K-Means).  
* **Temporal EHR Modeling:** Utilizes GRU-based Autoencoders to capture disease progression and clinical trajectories across multiple visits.  
* **Multi-modal Embedding:** Integrated support for Numerical (Autoencoders), Categorical (Med2Vec-style), and Textual (ClinicalBERT) data.  
* **Weighted Fusion:** Learnable importance weights for different clinical domains.

## **Repository Structure**

```
Cluster-Guided-Embedding/    
├── data/                   \# Dataset directory (Note: BioArc raw data is private)    
│   └── sample_patient.json \# Structure of expected input    
├── src/                    \# Source code directory    
│   ├── preprocessing/      \# Package for data cleaning and tokenization    
│   │   ├── preprocessor.py     \# BioArc specific preprocessing  
│   │   └── mimic_processor.py  \# MIMIC-IV preprocessing logic  
│   ├── clustering/         \# Package for feature grouping logic    
│   │   ├── feature_clustering.py  
│   │   └── mimic_validation.py \# Statistical validation of clusters 
│   ├── models/             \# Package for neural network architectures    
│   │   ├── embeddings.py       \# Proposed encoders  
│   │   └── baselines.py        \# Global Baseline implementation 
│   └── fusion/             \# Package for integration and weighting    
│       └── integration.py    
├── weights/                \# Pre-trained model weights (.pt files)    
├── train.py                \# Main training pipeline script    
├── test_similarity.py      \# Evaluation and metric calculation script    
├── test_mimic_validation.py\# External validation benchmark 
└── requirements.txt        \# Environment dependencies
```

## **Installation**

1. Clone the repository:

```bash
   git clone https://github.com/tanhaei/ClusterGuidedEmbedding.git 
   cd ClusterGuidedEmbedding
```

2. Install dependencies:

```bash
   pip install -r requirements.txt
```

## **How to Run (BioArc Cohort)**

### **1\. Feature Clustering & Training**

To run the full pipeline (preprocessing, clustering, and model training):

```bash
python train.py
```

### **2\. Evaluation**

To evaluate the model against expert-annotated patient pairs (Gold Standard):

```bash
python test_similarity.py
```

## **External Validation (MIMIC-IV)**

To address the concern regarding generalizability and reproducibility, we have provided the full pipeline for validation on the **MIMIC-IV dataset**.

### **1\. Preprocessing & Clinical Grouping**

We map raw MIMIC-IV features (Vitals, Labs, Meds, Notes) into 5 clinical clusters. The preprocessing script handles missing values, normalization, and temporal alignment.

```bash
python src/preprocessing/mimic_processor.py
```

### **2\. Preliminary Cluster Results**

Before training, the cohesion of clinical clusters is validated. This script calculates the **Silhouette Score** and **Calinski-Harabasz Index** to justify the choice of clusters as reported in the paper.

```bash
python src/clustering/mimic_validation.py
```

### **3\. Benchmarking: Global Baseline vs. Cluster-Guided**

We provide a comparative benchmark between the standard **Global Embedding** approach (no clustering) and our proposed **Cluster-Guided** framework.

```bash
python test_mimic_validation.py
```

| Method | Precision@10 | Recall@10 | F1-Score |
| :---- | :---- | :---- | :---- |
| Global Baseline | 0.74 | 0.69 | 0.71 |
| **Cluster-Guided (Proposed)** | **0.79** | **0.74** | **0.76** |


## **Pre-trained Weights**

Pre-trained weights for the Ophthalmic and Systemic clusters are provided in the weights/ directory. These weights were trained on the BioArc ophthalmology cohort (N=5,000).

## **Data Privacy**

The raw EHR data from the BioArc system is protected by patient privacy regulations. A sample\_patient.json is provided to illustrate the required schema. For external validation, we recommend using the MIMIC-IV dataset with our provided scripts.

## **Citation**

If you use this framework in your research, please cite:
