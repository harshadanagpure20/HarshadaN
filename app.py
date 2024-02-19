from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        preprocessed_text = preprocess_text(text)
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(preprocessed_text)
        if scores['compound'] > 0:
            sentiment = 'Positive'
        elif scores['compound'] < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        return render_template('results.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
