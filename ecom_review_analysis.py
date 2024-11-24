import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# File paths
input_file = 'reviews.csv'  # Replace with your input file path
output_file_sentiment = 'sentiment_analysis.csv'
output_file_duplicates = 'fake_reviews.csv'
output_file_recommendations = 'product_recommendations.csv'

# Load the data
df = pd.read_csv(input_file)

# Sentiment Analysis Function
def analyze_sentiment(review):
    sentiment = TextBlob(review).sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df['sentiment'] = df['text'].apply(analyze_sentiment)

# Fake Review Detection: Flagging duplicate reviews
df['is_duplicate'] = df.duplicated(subset=['text'], keep=False)

# Recommend Products Based on Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])
similarity = cosine_similarity(tfidf_matrix)

recommended_products_list = []
for index, row in df.iterrows():
    similar_indices = similarity[index].argsort()[-3:][::-1]  # Top 3 similar reviews
    similar_products = df.iloc[similar_indices]['product'].unique()
    recommended_products_list.append(", ".join(similar_products))

df['recommended_products'] = recommended_products_list

# Save Results
df[['review_id', 'product', 'text', 'sentiment']].to_csv(output_file_sentiment, index=False)
df[df['is_duplicate']].to_csv(output_file_duplicates, index=False)
df[['review_id', 'product', 'recommended_products']].to_csv(output_file_recommendations, index=False)

print("Analysis complete! Results saved to:")
print(f"1. Sentiment Analysis: {output_file_sentiment}")
print(f"2. Flagged Fake Reviews: {output_file_duplicates}")
print(f"3. Product Recommendations: {output_file_recommendations}")
