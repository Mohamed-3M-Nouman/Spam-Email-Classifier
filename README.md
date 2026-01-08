# 📧 AI Spam Email Classifier

A Machine Learning application built with **Python** and **Streamlit** to detect Spam emails using Natural Language Processing (NLP) techniques.

## 🧠 Model Overview
The project uses the **Multinomial Naive Bayes** algorithm, which is highly effective for text classification.
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text data into numerical vectors.
- **Accuracy:** The model provides a real-time accuracy score and classification report based on the testing dataset.

## 🚀 Features
- **Interactive Interface:** Built with Streamlit for a smooth user experience.
- **Real-time Prediction:** Enter any text to check if it's "Spam" or "Ham" (Safe).
- **Data Visualization:** Displays Confusion Matrix heatmap and dataset distribution graphs using Seaborn & Matplotlib.
- **Easy Launch:** Includes a `.bat` file for one-click execution on Windows.

## 🛠️ Technologies Used
- **Python** (Core Language)
- **Scikit-Learn** (Model Building & Evaluation)
- **Pandas & NumPy** (Data Manipulation)
- **Streamlit** (Web GUI)
- **Matplotlib & Seaborn** (Visualization)

## 📂 Project Structure
- `ML_spam_email.py`: The main application script.
- `preprocessed_data.csv`: The cleaned dataset used for training.
- `run_spam_app.bat`: Batch script to run the app instantly on Windows.

## 💻 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NoumanDev/Spam-Email-Classifier.git](https://github.com/NoumanDev/Spam-Email-Classifier.git)
Install dependencies:

Bash

pip install -r requirements.txt
Run the application:

Method 1 (Command Line):

Bash

streamlit run ML_spam_email.py
Method 2 (Windows): Double-click the run_spam_app.bat file.

👨‍💻 Author
Mohamed Nouman - AI & CS Student at Assiut University."# Spam-Email-Classifier" 
"# Spam-Email-Classifier" 
