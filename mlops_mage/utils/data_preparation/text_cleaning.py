import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('punkt')


# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Function for text cleaning
def clean_review(text):

    # Regular expression to detect URLs
    url_pattern = re.compile(r'(https?://\S+|www\.\S+)')

    # Replace URLs with 'URL' word
    text = url_pattern.sub('URL', text)

    # Remove HTML tags using BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and punctuation using regular expressions
    re_punctuations = re.compile('[%s]'%re.escape(string.punctuation))
    text = re_punctuations.sub(' ', text)

    # Remove numbers
    text = re.sub(r'[\d+]', '', text)

    # Replace multiple spaces with one single space
    text = re.sub(r'\s+', ' ', text)

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Remove stopwords and perform lemmatization
    cleaned_text = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Join the words back into a single string
    return ' '.join(cleaned_text)


# Function to clean review text column
def clean(df_in, text_col='text_'):
    '''
    This function takes a dataframe with review text column as input and 
    returns a dataframe with columns for cleaned text and its length.
    '''
    df_cleaned = df_in.copy()
    df_cleaned['cleaned_text'] = df_cleaned[text_col].apply(clean_review)
    df_cleaned['cleantext_length'] = df_cleaned['cleaned_text'].apply(len)

    return df_cleaned