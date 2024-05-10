import logging
import re
import requests
from dateutil.parser import parse
import spacy
import nltk
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import speech_recognition as sr
import pyttsx3
from ML import MLModel  # Importing the MLModel from ML.py

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model and NLTK resources safely
try:
    nlp = spacy.load('en_core_web_trf')
    nltk.download('punkt')
except Exception as e:
    logger.error(f"Failed to load NLP resources: {e}")
    raise SystemExit(e)

# Initialize the text-to-speech and speech recognition engines
try:
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
except Exception as e:
    logger.error(f"Failed to initialize speech engines: {e}")
    raise SystemExit(e)

# Create an instance of MLModel
ml_model = MLModel(logger)

def extract_entities(text):
    """Extract named entities using spaCy's NER."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def parse_dates(text):
    """Parse dates from text using spaCy's NER and dateutil."""
    dates = [ent.text for ent in nlp(text).ents if ent.label_ == 'DATE']
    return [parse(date, fuzzy=True).strftime('%Y-%m-%d') for date in dates if date]

def analyze_text(text):
    """Enhanced NLP processing with machine learning integration."""
    entities = extract_entities(text)
    sentiment = ml_model.analyze_sentiment(text)
    summary = ml_model.summarize_text(text)
    logger.info(f'Entities: {entities}, Sentiment: {sentiment}, Summary: {summary}')
    return entities, sentiment, summary

def scrape_web_page(url):
    """Scrape text from a webpage and analyze it."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = ' '.join(p.text for p in soup.find_all('p'))
        return analyze_text(text_content)
    except requests.RequestException as e:
        logger.error(f"Web scraping error: {e}")
        return None

def dynamic_response(text):
    """Generate responses based on simple pattern matching."""
    if re.search(r'\b(hello|hi|hey)\b', text, re.IGNORECASE):
        return "Hello! How can I assist you today?"
    elif re.search(r'how are you', text, re.IGNORECASE):
        return "I'm just a program, but thank you for asking!"
    return f"You said: {text}"

def listen_and_respond():
    """Listen to user speech and respond based on content analysis."""
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        logger.info("Listening for speech...")
        try:
            audio = recognizer.listen(source)
            speech_text = recognizer.recognize_google(audio)
            logger.info(f"Recognized speech: {speech_text}")
            response = dynamic_response(speech_text)
            engine.say(response)
            engine.runAndWait()
        except sr.UnknownValueError:
            engine.say("I didn't catch that. Could you please repeat?")
            engine.runAndWait()
        except sr.RequestError as e:
            engine.say("I'm having trouble with the speech service right now.")
            engine.runAndWait()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            engine.say("An error occurred. Please try again.")
            engine.runAndWait()

if __name__ == "__main__":
    while True:
        listen_and_respond()