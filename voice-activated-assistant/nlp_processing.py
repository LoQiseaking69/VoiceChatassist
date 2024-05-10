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

# Load spaCy model and NLTK resources
nlp = spacy.load('en_core_web_trf')
nltk.download('punkt')

# Initialize the text-to-speech and speech recognition
engine = pyttsx3.init()
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Create an instance of MLModel
ml_model = MLModel(logger)

def analyze_text(text):
    try:
        # Enhanced NLP processing with machine learning integration
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        sentiment = ml_model.analyze_sentiment(text)  # Analyze sentiment using ML
        summary = ml_model.summarize_text(text)  # Summarize text using ML
        
        # Log the analysis results
        logger.info(f'Entities: {entities}, Sentiment: {sentiment}, Summary: {summary}')
        
        return entities, sentiment, summary
    except Exception as e:
        logger.error(f"Error in text analysis: {e}")
        return [], None, None

def listen_and_respond():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            analysis_results = analyze_text(text)
            response = generate_response_based_on_analysis(analysis_results)  # Generate responses based on ML analysis
            engine.say(response)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in speech recognition or response generation: {e}")
            engine.say("Sorry, I didn't catch that. Could you repeat?")
            engine.runAndWait()

def generate_response_based_on_analysis(analysis_results):
    # This function would generate intelligent responses based on the analysis
    entities, sentiment, summary = analysis_results
    if sentiment and sentiment > 0.5:
        return "That sounds positive! How can I assist further?"
    else:
        return "I'm here to help. Tell me more."

# Main loop to handle interactions
if __name__ == "__main__":
    while True:
        listen_and_respond()