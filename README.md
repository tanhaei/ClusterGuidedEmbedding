# **Cluster-Guided Embedding for Patient Similarity Search in BioArc**

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
├── data/                   # Dataset directory (Note: BioArc raw data is private)  
│   └── sample_patient.json # Structure of expected input  
├── src/                    # Source code directory  
│   ├── preprocessing/      # Package for data cleaning and tokenization  
│   │   └── preprocessor.py  
│   ├── clustering/         # Package for feature grouping logic  
│   │   └── feature_clustering.py  
│   ├── models/             # Package for neural network architectures  
│   │   └── embeddings.py  
│   └── fusion/             # Package for integration and weighting  
│       └── integration.py  
├── weights/                # Pre-trained model weights (.pt files)  
├── train.py                # Main training pipeline script  
├── test_similarity.py      # Evaluation and metric calculation script  
└── requirements.txt        # Environment dependencies
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

## **How to Run**

### **1\. Feature Clustering & Training**

To run the full pipeline (preprocessing, clustering, and model training), execute:

```bash
python train.py
```

This script will:

* Process input features from data/.  
* Perform K-Means clustering to identify clinical groups.  
* Train **Temporal Autoencoders (GRU)** for each cluster and learn fusion weights. 
* Save weights to the weights/ directory.

### **2\. Evaluation**

To evaluate the model against expert-annotated patient pairs (Gold Standard):

```bash
python test_similarity.py
```

This will report **Precision, Recall, F1-Score, AUC-ROC, and MRR**, along with the average inference time per query.

## **Pre-trained Weights**

In compliance with reviewer recommendations, pre-trained weights for the Ophthalmic and Systemic clusters are provided in the weights/ directory. These weights were trained on the BioArc ophthalmology cohort (N=5,000).

## **Data Privacy**

The raw EHR data from the BioArc system is protected by patient privacy regulations. A sample_patient.json is provided to illustrate the required schema. For external validation, we recommend using the MIMIC-IV dataset with our adapted clustering logic.

## **Citation**

If you use this framework in your research, please cite:

