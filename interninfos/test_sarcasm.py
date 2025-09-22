from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

text = "I was deeply frustrated by how flawless everything turned out—staff too polite, service annoyingly quick, food irritatingly delicious, and atmosphere excessively perfect. I came ready to complain, but they robbed me of that joy. Honestly, it's unbearable how much I enjoyed the entire experience. Truly disappointing perfection."

scores = analyzer.polarity_scores(text)
print("Sarcastic positive review scores:")
print(scores)
print("Compound:", scores['compound'])
print("Classification:", "Positive" if scores['compound'] >= 0.05 else "Negative" if scores['compound'] <= -0.05 else "Neutral")
