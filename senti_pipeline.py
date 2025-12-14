import re
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# BASIC CLEANING

def clean_text_for_sentiment(text):
    """
    Removes noise but preserves all sentiment-relevant structure.
    """

    # Remove URLs (they add no emotional information)
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove emojis that may break tokenization 
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)

    # Remove punctuation except ! ? and apostrophes
    # These carry sentiment intensity and negation structure.
    text = re.sub(r"[^\w\s'!?]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# TOKENIZATION + OPTIONAL LEMMATIZATION

def tokenize_and_lemmatize(text, lemmatize=True):
    """
    Tokenizes text and optionally applies lemmatization.
    """

    tokens = word_tokenize(text)

    if lemmatize:
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]

    return tokens


# MAIN SENTIMENT PIPELINE

def pipeline_for_sentiment(text, 
                           lowercase=True,
                           lemmatize=True,
                           convert_diacritics=True):
    """
    Full preprocessing pipeline specifically optimized for sentiment analysis
    This pipeline:
    - keeps negations
    - keeps punctuation
    - keeps intensifiers
    - removes only true noise (URLs, emojis, weird characters)
    """

    text = clean_text_for_sentiment(text)

    if convert_diacritics:
        text = unidecode(text)

    if lowercase:
        text = text.lower()

    tokens = tokenize_and_lemmatize(text, lemmatize=lemmatize)

    return " ".join(tokens)
