import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# --- 1. إعداد الموديل والبيانات (من ملف ML_spam_email.py) ---
# تجاهل التحذيرات
warnings.filterwarnings('ignore', category=UserWarning)

# إخفاء التحذير الخاص بـ st.pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

# تحميل البيانات
try:
    df = pd.read_csv(R"preprocessed_data.csv")
except FileNotFoundError:
    st.error("خطأ: لم يتم العثور على ملف 'preprocessed_data.csv'. تأكد من وجود الملف في نفس المجلد.")
    st.stop() # إيقاف تشغيل التطبيق إذا لم يتم العثور على الملف

# إعادة تسمية الأعمدة (كما في الكود الأصلي)
df.rename(columns={'Category': 'target', 'Message': 'email'}, inplace=True)

# تحضير المتغيرات
X = df['email']
y = df['target']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', strip_accents='unicode', decode_error='ignore')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# تدريب الموديل
model_AI_NB = MultinomialNB()
model_AI_NB.fit(X_train_vec, y_train)

# التنبؤ (لإظهار المقاييس)
y_pred = model_AI_NB.predict(X_test_vec)

# --- 2. تعريف واجهة المستخدم الرسومية (الكود الجديد الخاص بك) ---

def predict_message():
    st.set_page_config(page_title="Spam Email Classifier", layout="centered")
    st.title("📩 Spam Email Classifier")
    st.write("Paste your email message below and classify it:")

    st.markdown("---")
    user_input = st.text_area("✉️ Enter your message here:", height=200)

    if st.button("Classify"):
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            # استخدام المتغيرات العالمية 'vectorizer' و 'model_AI_NB'
            vector = vectorizer.transform([user_input.lower()])
            prediction = model_AI_NB.predict(vector)[0]

            if prediction == 1:
                st.error("🔴 Prediction: SPAM")
            else:
                st.success("🟢 Prediction: HAM (Not spam)")

    # --- 3. عرض مقاييس أداء الموديل ---
    
    # استخدام المتغيرات العالمية 'y_test' و 'y_pred'
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
    # استخدام المتغير العالمي 'df'
    label_counts = df['target'].value_counts()
    label_names = ['Ham', 'Spam']

    fig2, ax2 = plt.subplots()
    ax2.pie(label_counts, labels=label_names, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
    ax2.axis('equal')
    st.pyplot(fig2)

# --- 4. تشغيل التطبيق ---
predict_message()

"""
         By ENG\ Mohamed Noaman 
               AI Engineer  
"""
