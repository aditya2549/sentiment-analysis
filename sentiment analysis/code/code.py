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