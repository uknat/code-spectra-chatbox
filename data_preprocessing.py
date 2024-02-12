from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_data(data):
    """
    Preprocess conversational data for NLP tasks.
    """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    data['processed'] = data['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x.lower()) if word not in stop_words and word.isalpha()]))
   
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['processed'])
    
    return vectors, vectorizer

vectors, vectorizer = preprocess_data(data)
