# 🧠 Mental Health sentiment analysis using NLP & Machine Learning  

## 🩺 About the Project  
Mental health plays a crucial role in overall well-being, yet many individuals struggle in silence.  
This project aims to identify early signs of **mental health issues** — such as **stress, depression, harassment, or bipolar disorder** — by analyzing user-generated text.  

Using **Natural Language Processing (NLP)** and **Machine Learning**, this model classifies text into mental health categories to support awareness and early intervention.

---

## 🚀 Project Overview  
The dataset (sourced from **Kaggle**) contains text samples labeled with mental states like:  
- **Normal**  
- **Stressed**  
- **Depressed**  
- **Bipolar**  
- **Harassed**

The project applies NLP techniques for text preprocessing and multiple ML models for classification.  
The performance of each model is compared to determine the best-performing one.

---

## 🧩 Features  

✅ **Text Preprocessing (NLP Pipeline)**  
- Tokenization, Lemmatization, Stopword Removal (using **NLTK**)  
- Lowercasing and noise removal (punctuation, URLs, emojis, etc.)  
- Feature extraction using **TF-IDF Vectorization**  

✅ **Machine Learning Models Implemented**  
- Decision Tree Classifier  
- Logistic Regression  
- Naïve Bayes Classifier  
- XGBoost Classifier  

✅ **Hyperparameter Tuning**  
- Utilized **GridSearchCV** and **RandomizedSearchCV** for parameter optimization  

✅ **Visualization & Insights**  
- Class distribution and text analytics  
- Word frequency visualization  
- Model performance comparison (accuracy, precision, recall, F1-score)  
- Confusion matrix visualization  

✅ **Evaluation Metrics**  
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Cross-validation performance  

---

## 🧠 Technologies & Libraries Used  

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **NLP Processing** | NLTK, Regex, TF-IDF |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Model Optimization** | GridSearchCV, RandomizedSearchCV |

---

## 🧪 Workflow Summary  

1. **Data Loading & Cleaning**  
   - Loaded dataset from Kaggle  
   - Handled missing values and class imbalance  

2. **Text Preprocessing**  
   - Tokenization → Stopword Removal → Lemmatization → TF-IDF Vectorization  

3. **Model Training**  
   - Trained Decision Tree, Logistic Regression, Naïve Bayes, and XGBoost classifiers  

4. **Evaluation**  
   - Used accuracy, precision, recall, and F1-score metrics  
   - Plotted confusion matrices for visual comparison  

5. **Hyperparameter Tuning**  
   - Optimized models using GridSearchCV / RandomizedSearchCV  

6. **Visualization**  
   - Analyzed class distribution and feature importance  

---


## 🔍 Future Improvements  

- Deploy the model using **Streamlit** or **Flask** for real-time predictions  
- Integrate **Deep Learning models (LSTM / BERT)** for better contextual understanding  
- Expand the dataset with more diverse mental health categories  
- Enable **real-time monitoring** from social media or chat data  

---





# Run the Jupyter Notebook
jupyter notebook mental_health_detection.ipynb
