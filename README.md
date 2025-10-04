
# 📩 Email/SMS Spam Classifier

## 🧠 Project Overview

This project is a **Machine Learning–based Email/SMS Spam Detection System** that classifies messages as **Spam** or **Not Spam**. It uses **Natural Language Processing (NLP)** techniques for text preprocessing and a **Multinomial Naive Bayes (MNB)** model for classification.

The model achieves **~97% accuracy** and **1.0 precision**, making it highly reliable for real-world spam detection tasks. The application is deployed as an interactive **Streamlit web app** where users can enter a message and instantly check whether it’s spam.

---

## ✨ Features

* 📉 **Label Encoding**: Encodes target labels (spam/ham) into numeric form
* 🔠 **Text Preprocessing**:

  * Convert text to **lowercase**
  * **Tokenization** – split text into words
  * **Remove special characters**, numbers, and punctuation
  * **Remove stopwords** (e.g., *is, of, the*)
  * **Stemming** using `PorterStemmer`
* 📊 **Data Visualization**:

  * **Seaborn** and **Matplotlib** for EDA
  * **Word Cloud** to visualize most frequent words in spam vs. non-spam
* 🔎 **Feature Extraction**:

  * **TF-IDF Vectorization** for text representation
* 🤖 **Machine Learning**:

  * **Multinomial Naive Bayes (MNB)** classifier
* ✅ **Performance**:

  * Precision: **1.0**
  * Accuracy: **~97%**
* 💾 **Model Serialization**:

  * Save trained model and vectorizer using **Pickle**
* 🌐 **Web Interface**:

  * Built with **Streamlit** for real-time predictions

---

## 🧰 Tech Stack

* 🐍 **Python**
* 📚 **NLTK** – NLP preprocessing
* 📊 **Matplotlib**, **Seaborn** – Data visualization
* ☁️ **WordCloud** – Text visualization
* 🧠 **Scikit-learn** – ML algorithms & metrics
* ✉️ **TF-IDF Vectorizer** – Text feature extraction
* 🤖 **Multinomial Naive Bayes (MNB)** – Classification
* 💾 **Pickle** – Model saving/loading
* 🌐 **Streamlit** – Web app development

---

## 📊 Workflow

### 1️⃣ Data Preprocessing

* Convert text to **lowercase**
* **Tokenize** text into words
* Remove **special characters**, **punctuation**, and **stopwords**
* Apply **stemming** to reduce words to their root form

### 2️⃣ Exploratory Data Analysis (EDA)

* Visualize data distribution using **Seaborn** and **Matplotlib**
* Generate **word clouds** for spam and non-spam messages

### 3️⃣ Feature Engineering

* Transform text into numerical vectors using **TF-IDF**

### 4️⃣ Model Training

* Train a **Multinomial Naive Bayes** classifier
* Evaluate using accuracy, precision, recall, and F1-score

### 5️⃣ Model Deployment

* Save the model and vectorizer with **Pickle**
* Build a **Streamlit** interface for real-time message classification

---

## 📈 Results

| Metric    | Score    |
| --------- | -------- |
| Precision | **1.0**  |
| Accuracy  | **~97%** |


---


### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at: `http://localhost:8501`

---

## 🔮 Future Work

* Deploy on **Streamlit Cloud** or **Hugging Face Spaces**
* Create a browser extension for live email filtering


---
