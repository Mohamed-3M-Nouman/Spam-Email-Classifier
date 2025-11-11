import streamlit as st
import joblib
import string
import re

# تحميل الموديل
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# دالة لتنظيف النص
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# واجهة Streamlit
st.title("📧 Spam Email Classifier")
st.write("تحقق مما إذا كانت الرسالة Spam أو Ham.")

email_text = st.text_area("اكتب الرسالة هنا:")

if st.button("تحليل"):
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    if prediction == 1:
        st.error("🚨 الرسالة Spam!")
    else:
        st.success("✅ الرسالة Ham (غير مزعجة).")
