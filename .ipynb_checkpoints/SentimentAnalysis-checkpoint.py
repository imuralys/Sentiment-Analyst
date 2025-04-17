import streamlit as st
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Tiêu đề ứng dụng
st.title('Sentiment Analysis with TextBlob, VADER and Hugging Face')

# Khởi tạo công cụ phân tích cảm xúc
analyzer = SentimentIntensityAnalyzer()
hf_pipeline = pipeline("sentiment-analysis")

# Hàm phân tích cảm xúc với TextBlob
def analyze_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'Positive', polarity
    elif polarity < -0.1:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

# Hàm phân tích cảm xúc với VADER
def analyze_vader(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound > 0.1:
        return 'Positive', score
    elif compound < -0.1:
        return 'Negative', score
    else:
        return 'Neutral', score
    
# Hàm phân tích cảm xúc với Hugging Face
def analyze_huggingface(text):
    result = hf_pipeline(text)[0]
    return result['label'], result['score']

# Phân tích cảm xúc cho văn bản nhập từ người dùng
st.header("Analyze Sentiment for a Custom Text")
with st.expander('Enter Text for Analysis'):
    custom_text = st.text_area("Type your text here:")
    
    if custom_text:
        # Phân tích với TextBlob
        tb_sentiment, tb_polarity = analyze_textblob(custom_text)
        st.subheader("TextBlob Analysis")
        st.write(f"Sentiment: {tb_sentiment}")
        st.write(f"Polarity Score: {tb_polarity:.2f}")
        
        # Phân tích với VADER
        vader_sentiment, vader_scores = analyze_vader(custom_text)
        st.subheader("VADER Analysis")
        st.write(f"Sentiment: {vader_sentiment}")
        st.write(f"Polarity Scores: {vader_scores}")

        # Phân tích với Hugging Face
        hf_sentiment, hf_score = analyze_huggingface(custom_text)
        st.subheader("HuggingFace Transformers Analysis")
        st.write(f"Sentiment: {hf_sentiment}")
        st.write(f"Polarity Scores: {hf_score:.2f}")

# Giao diện tải lên file CSV
st.header('Upload a CSV File for Analysis')
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Đọc file CSV
    df = pd.read_csv(uploaded_file)

    # Xem trước dữ liệu
    st.write("Preview of the uploaded dataset:")
    st.write(df.head())
    
    # Chọn cột chứa văn bản
    text_column = st.selectbox("Select the column containing text data", df.columns)
    
    if text_column:
        # Đảm bảo cột văn bản là kiểu chuỗi và xử lý giá trị NaN
        df[text_column] = df[text_column].fillna("").astype(str)
        
        # Thêm cột phân tích cảm xúc từ TextBlob
        #st.subheader('Sentiment Analysis using TextBlob')
        df['TextBlob Sentiment'], df['TextBlob Polarity'] = zip(*df[text_column].apply(analyze_textblob))
        
        # Thêm cột phân tích cảm xúc từ VADER
        #st.subheader('Sentiment Analysis using VADER')
        df['VADER Sentiment'], df['VADER Scores'] = zip(*df[text_column].apply(analyze_vader))

        # Thêm cột phân tích cảm xúc từ Hugging Face
        #st.subheader('Sentiment Analysis using Hugging Face Transformers')
        df['HF Sentiment'], df['HF Score'] = zip(*df[text_column].apply(analyze_huggingface))

        # Hiển thị kết quả
        st.write("Analyzed Dataset:")
        st.write(df.head())

        # Tải xuống kết quả
        st.download_button(
            label="Download Analyzed Data as CSV",
            data=df.to_csv(index=False),
            file_name="analyzed_sentiments.csv",
            mime="text/csv",
        )