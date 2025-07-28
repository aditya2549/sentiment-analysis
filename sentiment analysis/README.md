Customer Feedback Sentiment Analysis System

Project Overview
This project automates sentiment analysis of customer feedback collected from various channels, providing actionable insights and visualizations in PowerBI.

Dataset
The dataset consists of customer feedback entries labeled with their source (email, survey, web form).

Code
The Python script processes the feedback texts, computes sentiment polarity with TextBlob, and classifies each as Positive, Neutral, or Negative.

PowerBI Dashboard
Visualizes sentiment distribution over time and by categories, providing management with key insights.

 Workflow Diagram
1. Collect customer feedback data
2. Preprocess and clean text data
3. Apply sentiment analysis model
4. Export sentiment results to CSV
5. Import CSV to PowerBI
6. Create dashboards and visualizations

How to Run
- Place `customer_feedback.csv` in the same folder
- Run `sentiment_analysis.py` script
- Load `sentiment_results.csv` into PowerBI

Technologies Used
- Python, Pandas
- TextBlob for sentiment analysis
- PowerBI for dashboard visualization

Sample Code

import pandas as pd
from textblob import TextBlob

df = pd.read_csv('customer_feedback.csv')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars/numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

def get_sentiment(text):
return TextBlob(text).sentiment.polarity

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word not in stop_words)

df['clean_feedback'] = df['clean_feedback'].apply(remove_stopwords)


Apply sentiment analysis


df['sentiment_score'] = df['feedback'].apply(get_sentiment)

Label sentiment
def label_sentiment(score):
if score > 0.1:
return 'Positive'
elif score < -0.1:
return 'Negative'
else:
return 'Neutral'

df['sentiment_label'] = df['sentiment_score'].apply(label_sentiment)

df.to_csv('sentiment_results.csv', index=False)

print(df.head())

## Charting with Matplotlib

import matplotlib.pyplot as plt

df['sentiment_label'].value_counts().plot(kind='bar', color=['green', 'grey', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Feedbacks')
plt.show()









