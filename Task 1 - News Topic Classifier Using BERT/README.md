# 📰 News Classification using BERT

## 📌 Objective of the Task
The main goal of this project is to **classify news articles into predefined categories** using the **BERT (Bidirectional Encoder Representations from Transformers)** model.  
The dataset contains news headlines and descriptions, and the task is to train a transformer-based model that can accurately predict the category of a given news item.

---

## ⚙️ Methodology / Approach
1. **Data Preparation**
   - Used the **News Category Dataset v2**.
   - Combined headlines with short descriptions to enrich the text representation.
   - Cleaned category labels (e.g., merging `THE WORLDPOST` and `WORLDPOST`).

2. **Preprocessing**
   - Tokenized text using **HuggingFace BERT tokenizer**.
   - Truncated long sequences to fit the BERT model’s maximum input length.
   - Encoded target labels using `LabelEncoder`.

3. **Modeling**
   - Used **BERT** as the base model for text classification.
   - Implemented training on **TPUs (via Kaggle)** for efficiency.
   - Applied TensorFlow distribution strategies for scalability.

4. **Training**
   - Split data into **training and testing sets**.
   - Fine-tuned BERT with a classification head.
   - Optimized using Adam optimizer and categorical cross-entropy loss.

5. **Evaluation**
   - Measured accuracy on the test set.
   - Generated a **confusion matrix** to visualize misclassifications.

---

## 📊 Key Results / Observations
- Achieved an **accuracy of ~70.45%** on the test dataset.
- Observations:
  - Headlines significantly improved classification when combined with descriptions.
  - Some categories with overlapping content (e.g., `WORLDPOST` vs. `POLITICS`) were harder to distinguish.
  - The confusion matrix shows that misclassifications often occurred between semantically close categories.
- The model demonstrates that **transformer-based models like BERT are highly effective for multi-class text classification** tasks, even on noisy real-world datasets.

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/news-classification-bert.git
   cd news-classification-bert
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook news-classification-using-bert.ipynb
   ```

---

## 📂 Files

* `news-classification-using-bert.ipynb` → Jupyter Notebook containing code and experiments.
* `README.md` → Project documentation.