import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Spam Email Classifier", layout="centered")
warnings.filterwarnings("ignore", category=UserWarning)
@st.cache_resource
def train_model():
    df = pd.read_csv(R"preprocessed_data.csv")
    df.rename(columns={'Category': 'target', 'Message': 'email'}, inplace=True)
    df["email"] = df["email"].str.lower()
    
    x = df["email"]
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_vec, y_train)
    
    return model, vectorizer, x_test_vec, y_test, df

model_AI_NB, vectorizer, x_test_vec, y_test, df = train_model()
y_pred = model_AI_NB.predict(x_test_vec)

y_pred = model_AI_NB.predict(x_test_vec)

# حفظ الموديل والمتجه (vectorizer)
joblib.dump(model_AI_NB, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved successfully!")


# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

def test_by_user():
    user_input = input("Enter your email message: ")
    user_input = user_input.strip().lower()
    user_input = vectorizer.transform([user_input][0:])
    prediction = model_AI_NB.predict(user_input)
    print("🔴 SPAM" if prediction else "🟢 HAM (Not spam)")
# test_by_user()


########GUI_by_streamlit##################
def predict_message():
    # st.set_page_config(page_title="Spam Email Classifier", layout="centered")
    st.title("📩 Spam Email Classifier")
    st.write("Paste your email message below and classify it:")

    st.markdown("---")
    user_input = st.text_area("✉️ Enter your message here:", height=200)

    if st.button("Classify"):
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            vector = vectorizer.transform([user_input.lower()])
            prediction = model_AI_NB.predict(vector)[0]

            if prediction == 1:
                st.error("🔴 Prediction: SPAM")
            else:
                st.success("🟢 Prediction: HAM (Not spam)")

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.markdown(f"### ✅ Model Accuracy: **{acc:.2%}**")

    st.markdown("### 📊 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
    st.pyplot(fig)

    st.markdown("### 🧾 Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format(precision=2))
    st.markdown("### 📈 Dataset Distribution (Ham vs Spam)")
    label_counts = df['target'].value_counts()
    label_names = ['Ham', 'Spam']

    fig2, ax2 = plt.subplots()
    ax2.pie(label_counts, labels=label_names, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
    ax2.axis('equal')
    st.pyplot(fig2)


predict_message()

"""
         By ENG\ Mohamed Noaman 
              AI Engineer  
"""