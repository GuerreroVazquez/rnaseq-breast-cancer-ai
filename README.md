# 🧬 AI-Powered Breast Cancer Treatment Recommender

This repository contains the code and resources for a prototype bioinformatics pipeline developed for a postdoctoral research interview in **Bioinformatics, AI & Software Engineering**.

## 🎯 Purpose

To analyze RNA-seq gene expression data from breast cancer patients (GSE96058) and build an **AI-enhanced, end-to-end system** that generates **personalized treatment recommendations** based on **predicted survival outcomes**.

This project combines **bioinformatics, machine learning, deep learning, and large language model (LLM) orchestration** into a modular and interpretable framework designed to support **real-time, code-less clinical decision-making**.

---

## ⚙️ Key Features

- 🔄 **Preprocessing** of gene expression and clinical metadata (imputation based on SCAN-B estimates).
- 🧠 **Feature selection** using LASSO regression to reduce dimensionality.
- 🔍 **Exploratory analysis** with KNN clustering to confirm PAM50 subtypes.
- 🧬 **Autoencoder embedding** of gene expression to capture nonlinear biological structure.
- 📈 **Survival modeling** with Elastic Net-regularized Cox proportional hazards models.
- 💊 **Treatment outcome prediction** for multiple therapy combinations (endocrine, chemotherapy, both, or neither).
- 🤖 **LLM orchestration using LangGraph**, including:
  - `Master Node`: Interprets and routes user queries.
  - `Model Node`: Runs ML pipelines and returns predictions.
  - `RAG Node`: Retrieves relevant patient-level data.
  - `Literature Node`: Extracts evidence from biomedical publications.
- 🌐 **Streamlit app** for interactive, user-friendly recommendation interface.

---

## 🚧 Limitations

- Models are **prototypes** developed under time constraints for evaluation purposes.
- **No external validation** performed; results may not generalize.
- **Survival bias** may exist due to reliance on follow-up-based endpoints.
- Further development needed for robustness, explainability, and ethical integration.

---

## 📚 Dataset

- **Source**: [GSE96058 - NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058)
- **Description**: RNA-seq gene expression and clinical annotations for thousands of breast cancer patients.

---

## 🛠️ Stack

- **Languages**: Python
- **Libraries**: scikit-learn, pandas, lifelines, tensorflow/keras, shap, Streamlit
- **AI/LLM tools**: LangGraph, Gemini, Retrieval-Augmented Generation (RAG)
- **Data handling**: GEOquery, custom preprocessing scripts

---
  <h2>📦 Installation</h2>

## 1. Clone the repository
git clone https://github.com/GuerreroVazquez/rnaseq-breast-cancer-ai.git
cd rnaseq-breast-cancer-ai

## 2. Switch to the development branch
git checkout dev

## 3. (Optional) Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## 4. Install dependencies
pip install -r requirements.txt

## 5. Verify installation
python scripts/your_script.py --help


<h2>📁 Project Structure</h2>
  <ul>
    <li><code>data/</code>: Contains raw and cleaned RNA-seq data</li>
    <li><code>notebooks/</code>: Jupyter notebooks for preprocessing, analysis, and model evaluation</li>
    <li><code>src/</code>: Python scripts for preprocessing and modeling</li>
    <li><code>app/</code>: Streamlit app interface files</li>
    <li><code>models/</code>: (To be used) Saved AI/ML models</li>
    <li><code>requirements.txt</code>: Python dependencies list</li>
    <li><code>.gitignore</code>: Files and folders to exclude from version control</li>
    <li><code>README.html</code>: Project documentation (this file)</li>
  </ul>

  <h2>🔧 Dependencies</h2>
  <p>This project requires Python 3.8+ and uses:</p>
  <ul>
    <li><code>pandas</code></li>
    <li><code>numpy</code></li>
    <li><code>matplotlib</code></li>
    <li><code>scikit-learn</code></li>
    <li><code>seaborn</code></li>
    <li><code>jupyter</code></li>
    <li><code>streamlit</code></li>
  </ul>
  
