# E-Commerce Review Analysis Script  

This repository contains a Python script designed to help e-commerce businesses analyze customer reviews using AI techniques. The script performs:  

1. **Sentiment Analysis**: Categorizes reviews as Positive, Negative, or Neutral.  
2. **Fake Review Detection**: Flags duplicate reviews as potential spam.  
3. **Product Recommendations**: Suggests similar products based on review content.  

---

## Features  

- **Sentiment Analysis**: Uses TextBlob to evaluate customer reviews.  
- **Duplicate Review Detection**: Identifies spammy or duplicate content.  
- **TF-IDF Similarity**: Recommends products based on review similarity.  

---

## Requirements  

Install the following Python libraries before running the script:  

```bash
pip install pandas textblob scikit-learn
