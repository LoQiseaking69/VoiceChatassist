import asyncio
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from cryptography.fernet import Fernet
from database_interaction import ChatDatabase
from ml import MLModel
from speech_recognition import recognizer, microphone
from utilities import CustomLogger, EncryptedFileHandler
from nlp_processing import listen_and_respond, analyze_text

# Constants
LOG_FILE = 'app.log'
DATABASE_FILE = 'chat_memory.db'
ENCRYPTION_KEY_FILE = 'encryption_key.key'

# Function to retrieve the encryption key
def retrieve_encryption_key():
    try:
        with open(ENCRYPTION_KEY_FILE, 'rb') as key_file:
            key = key_file.read()
    except FileNotFoundError:
        raise FileNotFoundError("Encryption key file not found.")
    return key

# Initialize the logger with encryption
encryption_key = retrieve_encryption_key()
logger = CustomLogger(LOG_FILE, encryption_key=encryption_key)

# Initialize the ML model
ml_model = MLModel(logger)

# Initialize the database connection
db = ChatDatabase(DATABASE_FILE)

class VoiceAssistantApp(App):
    def build(self):
        # Ensure the database is initialized asynchronously
        asyncio.create_task(self.init_db_async())

        # Setup UI components
        self.label = Label(text='Speak into your microphone and the assistant will respond.')
        self.button = Button(text='Start Listening', on_press=self.listen)
        self.response_label = Label(text='Response will appear here...')

        # Layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        layout.add_widget(self.label)
        layout.add_widget(self.button)
        layout.add_widget(self.response_label)

        return layout

    async def init_db_async(self):
        try:
            await db.init_db()
        except Exception as e:
            logger.log_error(f"Error initializing database: {e}")

    async def process_audio(self):
        # Capture and process speech using the integrated NLP processing function
        text = listen_and_respond()
        if text:
            # Analysis and response generation based on the text processed
            analysis_results = analyze_text(text)
            entities, sentiment, summary = analysis_results
            logger.log_info(f"Processed interaction: Text: '{text}', Entities: {entities}, Sentiment: {sentiment}, Summary: {summary}")
            await db.update_response_frequency(summary)
            response = f"Entities: {entities}\nSentiment: {sentiment}\nSummary: {summary}"
            self.response_label.text = response

    def listen(self, instance):
        # Start listening to the microphone and process audio
        asyncio.create_task(self.process_audio())

    def on_stop(self):
        # Cleanly close the database connection when the application is closed
        asyncio.run(db.close())

if __name__ == '__main__':
    VoiceAssistantApp().run()