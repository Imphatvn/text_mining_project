import re
import nltk
import pandas as pd
import numpy as np
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A robust text preprocessor compatible with Scikit-Learn Pipelines.
    Incorporates regex logic from pipeline_v2.
    """
    def __init__(self, 
                 no_emojis=True, 
                 no_hashtags=True, 
                 hashtag_retain_words=True,
                 no_urls=True,
                 no_punctuation=True,
                 lowercase=True, 
                 lemmatize=True,
                 no_stopwords=True,
                 custom_stopwords=None):
        
        self.no_emojis = no_emojis
        self.no_hashtags = no_hashtags
        self.hashtag_retain_words = hashtag_retain_words
        self.no_urls = no_urls
        self.no_punctuation = no_punctuation
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.no_stopwords = no_stopwords
        self.custom_stopwords = set(custom_stopwords) if custom_stopwords else set()
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Handle pandas Series
        if isinstance(X, pd.Series):
            return X.apply(self._process_single_text).tolist()
        return [self._process_single_text(text) for text in X]

    def _regex_cleaner(self, raw_text):
        # Patterns extracted from your pipeline_v2.py
        hashtags_at_pattern = r"([#\@@\u0040\uFF20\uFE6B])"
        hashtags_ats_and_word_pattern = r"([#@]\w+)"
        emojis_pattern = r"([\u2600-\u27FF])"
        url_pattern = r"(?:\w+:\/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?"
        punctuation_pattern = r"[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u007C\u2010-\u2028\ufeff`]+"
        apostrophe_pattern = r"'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'"
        newline_pattern = r"(\n)"

        clean_text = str(raw_text)
        
        if self.no_emojis:
            clean_text = re.sub(emojis_pattern, "", clean_text)
            
        if self.no_hashtags:
            if self.hashtag_retain_words:
                clean_text = re.sub(hashtags_at_pattern, "", clean_text)
            else:
                clean_text = re.sub(hashtags_ats_and_word_pattern, "", clean_text)

        if self.no_urls:
            clean_text = re.sub(url_pattern, "", clean_text)
            
        if self.no_punctuation:
            clean_text = re.sub(punctuation_pattern, "", clean_text)
            clean_text = re.sub(apostrophe_pattern, "", clean_text)

        # Always clean newlines
        clean_text = re.sub(newline_pattern, " ", clean_text)
        
        return clean_text

    def _process_single_text(self, text):
        # 1. Regex Cleaning
        text = self._regex_cleaner(text)
        
        # 2. Unicode Normalization
        text = unidecode(text)

        # 3. Tokenization
        tokens = word_tokenize(text)

        # 4. Contraction Fixing (from your v2 pipeline)
        tokens = [re.sub(r"'m", "am", t) for t in tokens]
        tokens = [re.sub(r"n't", "not", t) for t in tokens]
        tokens = [re.sub(r"'s", "is", t) for t in tokens]

        # 5. Filtering & Lemmatization
        clean_tokens = []
        for token in tokens:
            # Lowercase
            if self.lowercase:
                token = token.lower()

            # Stopwords
            if self.no_stopwords:
                if token in self.stop_words or token in self.custom_stopwords:
                    continue
            
            # Lemmatize (Nouns, Verbs, Adjectives)
            if self.lemmatize:
                # Try converting to verb, then noun
                token = self.lemmatizer.lemmatize(token, pos='v')
                token = self.lemmatizer.lemmatize(token, pos='n')
                
            clean_tokens.append(token)
            
        return " ".join(clean_tokens)

def aggregate_reviews(df, group_col='title', text_col='raw_text', label_col='categoryName'):
    """
    Creativity/Ambition Helper:
    Groups reviews by restaurant (title) to create a larger text corpus per entity.
    """
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' missing. Did you drop it? Reload data without dropping '{group_col}'.")
        
    # Group by restaurant and join all reviews into one string
    # We aggregate the label too (assuming one restaurant has one label)
    df_grouped = df.groupby([group_col, label_col])[text_col].apply(lambda x: ' '.join(x)).reset_index()
    return df_grouped