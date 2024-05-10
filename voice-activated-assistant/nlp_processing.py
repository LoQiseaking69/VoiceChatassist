import spacy
import nltk
from nltk.tokenize import word_tokenize
import speech_recognition as sr
import pyttsx3
from ML import MLModel  # Importing the MLModel from ML.py
import logging

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model and NLTK resources safely
try:
    nlp = spacy.load('en_core_web_trf')
    nltk.download('punkt')
except Exception as e:
    logger.error(f"Failed to load NLP libraries: {e}")
    raise SystemExit(e)  # Exit if NLP libraries cannot be loaded

# Initialize the text-to-speech and speech recognition
try:
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
except Exception as e:
    logger.error(f"Failed to initialize speech engines: {e}")
    raise SystemExit(e)

# Create an instance of MLModel
ml_model = MLModel(logger)

def analyze_text(text):
    """Enhanced NLP processing with machine learning integration."""
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        sentiment = ml_model.analyze_sentiment(text)  # Analyze sentiment using ML
        summary = ml_model.summarize_text(text)  # Summarize text using ML
        
        logger.info(f'Entities: {entities}, Sentiment: {sentiment}, Summary: {summary}')
        return entities, sentiment, summary
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        return [], None, None

def listen_and_respond():
    """Handles speech recognition and response generation."""
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        logger.info("Listening for speech...")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            logger.info(f"Recognized text: {text}")
            analysis_results = analyze_text(text)
            response = generate_response_based_on_analysis(analysis_results)
            engine.say(response)
            engine.runAndWait()
        except sr.UnknownValueError:
            engine.say("Sorry, I didn't catch that. Could you repeat?")
            engine.runAndWait()
        except sr.RequestError as e:
            logger.error(f"Speech service error: {e}")
            engine.say("There was an error with the speech service.")
            engine.runAndWait()
        except Exception as e:
            logger.error(f"General error in speech recognition: {e}")
            engine.say("An error occurred. Please try again.")
            engine.runAndWait()

def generate_response_based_on_analysis(analysis_results):
    """Generate intelligent responses based on the analysis."""
    entities, sentiment, summary = analysis_results
    if sentiment is not None and sentiment > 0.5:
        return "That sounds positive! How can I assist further?"
    else:
        return "I'm here to help. Tell me more."

# Main loop to handle interactions
if __name__ == "__main__":
    logger.info("Starting the voice assistant...")
    try:
        while True:
            listen_and_respond()
    except KeyboardInterrupt:
        logger.info("Voice assistant terminated by user.")
        engine.stop()