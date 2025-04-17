import streamlit as st
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Khởi tạo VADER
analyzer = SentimentIntensityAnalyzer()

# Khởi tạo mô hình Hugging Face
huggingface_model = pipeline("sentiment-analysis")

# Tiêu đề ứng dụng
st.title("Ứng Dụng Phân Tích Cảm Xúc")

# Nhập văn bản trực tiếp
st.header("Nhập văn bản để phân tích")
user_input = st.text_area("Nhập văn bản của bạn ở đây:")

if st.button("Phân tích"):
    if user_input:
        # Phân tích bằng TextBlob
        textblob_result = TextBlob(user_input).sentiment

        # Phân tích bằng VADER
        vader_result = analyzer.polarity_scores(user_input)

        # Phân tích bằng Hugging Face
        huggingface_result = huggingface_model(user_input)

        # Hiển thị kết quả
        st.subheader("Kết quả phân tích cảm xúc:")
        st.write(f"**TextBlob:** Polarity: {textblob_result.polarity}, Subjectivity: {textblob_result.subjectivity}")
        st.write(f"**VADER:** {vader_result}")
        st.write(f"**Hugging Face:** {huggingface_result[0]}")

# Tải lên tệp CSV
st.header("Tải lên tệp CSV để xử lý hàng loạt")
uploaded_file = st.file_uploader("Chọn tệp CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dữ liệu đã tải lên:")
    st.write(df.head())

    # Phân tích cảm xúc cho từng hàng
    results = []
    for index, row in df.iterrows():
        text = row['Review']  # Giả sử cột chứa văn bản là 'Review'
        textblob_result = TextBlob(text).sentiment
        vader_result = analyzer.polarity_scores(text)
        huggingface_result = huggingface_model(text)

        results.append({
            "Review": text,
            "TextBlob Polarity": textblob_result.polarity,
            "TextBlob Subjectivity": textblob_result.subjectivity,
            "VADER": vader_result,
            "Hugging Face": huggingface_result[0]
        })

    # Chuyển đổi kết quả thành DataFrame
    results_df = pd.DataFrame(results)

    # Hiển thị kết quả
    st.subheader("Kết quả phân tích hàng loạt:")
    st.write(results_df)

    # Tính toán các chỉ số
    if 'Sentiment' in df.columns:  # Giả sử cột nhãn thực tế là 'Sentiment'
        true_sentiment = df['Sentiment'].apply(lambda x: 'POSITIVE' if x > 0 else 'NEGATIVE' if x < 0 else 'NEUTRAL')
        predicted_sentiment = results_df['Hugging Face'].apply(lambda x: x['label'])

        accuracy = accuracy_score(true_sentiment, predicted_sentiment)
        precision = precision_score(true_sentiment, predicted_sentiment, average='weighted', zero_division=0)
        recall = recall_score(true_sentiment, predicted_sentiment, average='weighted', zero_division=0)
        f1 = f1_score(true_sentiment, predicted_sentiment, average='weighted', zero_division=0)

        # Hiển thị các chỉ số
        st.write(f"Độ chính xác (Accuracy): {accuracy:.2f}")
        st.write(f"Độ chính xác (Precision): {precision:.2f}")
        st.write(f"Độ nhạy (Recall): {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

    # Tùy chọn tải xuống tệp kết quả
    csv = results_df.to_csv(index=False)
    st.download_button("Tải xuống tệp kết quả", csv, "results.csv", "text/csv") 