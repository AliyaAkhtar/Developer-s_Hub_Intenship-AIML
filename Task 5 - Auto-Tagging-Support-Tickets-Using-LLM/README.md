# Support Ticket Classification System

## Objective
This project automates the classification of IT support tickets into predefined categories, enabling faster routing and resolution of technical issues. The system helps support teams prioritize and manage incoming tickets more efficiently.

## Methodology / Approach

### 1. Data
- Dataset Source: [Kaggle](https://www.kaggle.com/datasets/giridharkrishnagiri/support-tickets-data)
- Utilizes a dataset of 21 historical support tickets with manually labeled categories (`dataset/support_ticket_data.csv`).
- Categories: Connectivity Issue, Hardware Malfunction, Data Recovery, Battery Issue, Account Access, Performance Issue, Software Issue.

### 2. Models Implemented
1. **Fine-tuned Transformer Model**
   - Uses a pre-trained DistilBERT model (`distilbert-base-uncased`) fine-tuned on the support ticket dataset in `ticket_classifier.ipynb`.
   - Saved to `./fine_tuned_model` and used for accurate classification of common ticket types, though performance is limited by small dataset size (accuracy: 0.4).

2. **FLAN-T5 Few-shot Learning**
   - Implements a few-shot learning approach using the pre-trained FLAN-T5 model (`google/flan-t5-base`).
   - Uses prompt engineering with example tickets to classify without fine-tuning, suitable for rare or new ticket types (accuracy: 0.71 on non-example tickets).
   - **Note**: Zero-shot classification is implemented separately using BART (`facebook/bart-large-mnli`) with high accuracy (0.90).

3. **Zero-shot Classification**
   - Uses BART (`facebook/bart-large-mnli`) for zero-shot classification, predicting categories without training data.
   - Achieves high accuracy (0.90) due to robust pre-trained knowledge.

### 3. Application
- Built with Streamlit (`app.py`) for an interactive web interface.
- Allows users to:
  - Input new support tickets via a text area.
  - Choose between Fine-tuned (DistilBERT) or FLAN-T5 Few-shot classification methods.
  - View top 3 predicted categories with confidence scores (Fine-tuned) or priority labels (FLAN-T5).
  - Browse and load sample tickets from the dataset, populating the text area for classification.

## Key Results / Observations

### Model Performance
- **Zero-shot (BART)**: Achieves highest accuracy (0.90) due to robust pre-trained knowledge, effective for most ticket types.
- **FLAN-T5 Few-shot**: Performs well (top-1 accuracy: 0.71) on non-example tickets, especially for rare categories like Battery Issue or Account Access.
- **Fine-tuned (DistilBERT)**: Lower accuracy (0.4) due to small dataset (21 tickets) and imbalanced classes (e.g., some categories have only one sample).
- **Model Selection**: Users can select between Fine-tuned and FLAN-T5 Few-shot, offering flexibility for different use cases.

## Getting Started

### Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
1. Run `ticket_classifier.ipynb` to fine-tune the DistilBERT model and save it to `./fine_tuned_model`.
2. Launch the Streamlit app:
```bash
streamlit run app.py
```