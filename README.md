# Fake News Detection System

This project is a machine learning-based web application designed to classify news articles as either **real** or **fake** using natural language processing (NLP) techniques. Built during an academic group project, this solution helps combat misinformation by leveraging supervised learning and text classification methods.

---

## 🚀 Features

- Predicts the authenticity of news articles using pre-trained ML models
- Text preprocessing using NLP techniques (tokenization, stop-word removal, TF-IDF)
- Achieved **95% classification accuracy**
- Web interface for input and output interaction
- Interpretable model results with minimal latency

---

## 🔧 Technologies Used

- **Language:** Python  
- **Libraries/Frameworks:** Scikit-learn, Pandas, NumPy, NLTK  
- **NLP:** TF-IDF Vectorizer, Tokenization, Stopword Filtering  
- **Modeling:** Logistic Regression / Naive Bayes / SVM (selectable during training)  
- **IDE:** Jupyter Notebook  
- **Visualization:** Matplotlib, Seaborn  
- **Version Control:** Git

---

## 📊 Dataset

The dataset was obtained from a **public open-source dataset** containing labeled real and fake news articles. Each entry includes a news title and full text.

---

## 🛠️ How It Works

1. **Text Preprocessing**: Clean and normalize text (lowercase, remove punctuation, etc.)
2. **Feature Extraction**: Convert text to numerical format using TF-IDF
3. **Model Training**: Fit classification model on labeled dataset
4. **Prediction**: User submits article text → returns predicted label (Real or Fake)

---

## 🧪 Model Evaluation

- Accuracy: 95%
- Metrics: Confusion Matrix, Precision, Recall, F1 Score
- Models Compared: Naive Bayes, Logistic Regression, SVM
- Best performing model: Naive Bayes (for balance between speed and accuracy)

---

## 📂 Project Structure

├── data/ # Raw and cleaned dataset files
├── notebooks/ # Jupyter Notebooks for EDA and model training
├── src/ # Scripts for preprocessing, training, and evaluation
├── app/ # Web interface code (if applicable)
└── README.md # Project documentation


---

## 🧑‍🤝‍🧑 Team Contribution

This project was completed as part of a **student group assignment**.  
**My role:** Led model training, feature engineering, and performance evaluation.

---

## 📌 Future Improvements

- Add deep learning support (e.g., LSTM, BERT)
- Integrate a Flask-based web app for real-time predictions
- Include interpretability features like SHAP/LIME

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 📬 Contact

For questions or collaboration:
**Vishwa Patel**  
📧 vishwapatel2103@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/vishwapatel2103) | [GitHub](https://github.com/Vishwa-patel21)
