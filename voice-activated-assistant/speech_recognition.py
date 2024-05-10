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

# Load spaCy model for English
nlp = spacy.load('en_core_web_trf')

# Download NLTK resources
nltk.download('punkt')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the speech recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Create an instance of MLModel
ml_model = MLModel(logger)

def extract_entities(text):
    # Extract named entities using spaCy's NER
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def parse_dates(text):
    # Parse dates using spaCy's NER and fallback to dateutil.parser
    doc = nlp(text)
    dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
    parsed_dates = []
    for date in dates:
        try:
            parsed_date = parse(date, fuzzy=True)
            parsed_dates.append(parsed_date.strftime('%Y-%m-%d'))
        except ValueError:
            logging.warning(f'Could not parse date: {date}')
    return parsed_dates

def analyze_text(text):
    # Enhanced NLP processing with machine learning integration
    entities = extract_entities(text)
    logging.info(f'Named entities: {entities}')
    
    parsed_dates = parse_dates(text)
    logging.info(f'Parsed dates: {parsed_dates}')
    
    tokens = word_tokenize(text)
    
    greeting_patterns = [r'\b(?:hello|hi|hey)\b', r'^(?:[Hh]ello|[Hh]i|[Hh]ey)[,.!?\s]+']
    for pattern in greeting_patterns:
        if re.search(pattern, text):
            logging.info('Found a greeting')
            return 'Found a greeting'
    
    doc = nlp(text)
    
    for token in doc:
        if token.lower_ == 'weather':
            weather_inquiry_patterns = [r'\b(?:like|forecast|temperature)\b', r'(?:like|forecast|temperature)[,.!?\s]+']
            for child in token.children:
                for pattern in weather_inquiry_patterns:
                    if re.search(pattern, child.text, re.IGNORECASE):
                        logging.info('Found a weather inquiry')
                        return 'Found a weather inquiry'
    
    return 'No matches found.'

def scrape_web_page(input_text):
    try:
        if input_text.startswith("http://") or input_text.startswith("https://"):
            response = requests.get(input_text)
            response.raise_for_status()  
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text_content = '\n'.join(paragraph.text for paragraph in paragraphs)
            result = analyze_text(text_content)
            return result
        else:
            return analyze_text(input_text)
    
    except Exception as e:
        logging.error(f"An error occurred while processing the input: {str(e)}")
        return 'Error: Could not process the input.'

def dynamic_response(text):
    pattern_greeting = re.compile(r'hello|hi|hey', re.IGNORECASE)
    pattern_question = re.compile(r'how are you', re.IGNORECASE)
    
    if pattern_greeting.search(text):
        return "Hello! How can I assist you today?"
    elif pattern_question.search(text):
        return "I'm just a computer program, but thank you for asking!"
    else:
        return None

def listen_and_respond():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        logging.info("Listening for speech...")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            logging.info(f"Recognized speech: {text}")
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
            response = dynamic_response(clean_text)
            if response:
                engine.say(response)
            else:
                engine.say(f"You said: {clean_text}")
            engine.runAndWait()
            return clean_text
        except sr.UnknownValueError:
            logging.error("Speech recognition could not understand audio")
            engine.say("I didn't catch that. Could you please repeat?")
            engine.runAndWait()
        except sr.RequestError as e:
            logging.error(f"Speech recognition service error: {e}")
            engine.say("I'm having trouble with the speech service right now.")
            engine.runAndWait()

if __name__ == "__main__":
    while True:
        listen_and_respond()