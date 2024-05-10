import asyncio
import toga
from toga.style import Pack
from toga.style.pack import COLUMN
from cryptography.fernet import Fernet
from database_interaction import ChatDatabase
from ml import MLModel
from speech_recognition import recognizer, microphone
from utilities import CustomLogger, EncryptedFileHandler
from nlp_processing import listen_and_respond, analyze_text

# Constants
LOG_FILE = 'app.log'
DATABASE_FILE = 'chat_memory.db'

# Function to generate or retrieve the encryption key
def generate_or_retrieve_key():
    # Check if a key exists in a file
    try:
        with open('encryption_key.key', 'rb') as key_file:
            key = key_file.read()
    except FileNotFoundError:
        # Generate a new key and save it to a file
        key = Fernet.generate_key()
        with open('encryption_key.key', 'wb') as key_file:
            key_file.write(key)
    return key

# Initialize the logger with encryption
encryption_key = generate_or_retrieve_key()
logger = CustomLogger(LOG_FILE, encryption_key=encryption_key)

# Initialize the ML model
ml_model = MLModel(logger)

# Initialize the database connection
db = ChatDatabase(DATABASE_FILE)

class VoiceAssistantApp(toga.App):
    def startup(self):
        # Ensure the database is initialized asynchronously
        self.init_db_async()

        # Setup main window
        self.main_window = toga.MainWindow(title="Voice-Activated Personal Assistant")
        self.main_window.size = (640, 480)

        # Setup UI components
        self.label = toga.Label('Speak into your microphone and the assistant will respond.')
        self.button = toga.Button('Start Listening', on_press=self.listen)
        self.response_label = toga.Label('Response will appear here...')

        # Layout
        box = toga.Box(children=[self.label, self.button, self.response_label],
                       style=Pack(direction=COLUMN, padding=10))
        self.main_window.content = box
        self.main_window.show()

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

    def listen(self, widget):
        # Start listening to the microphone and process audio
        asyncio.create_task(self.process_audio())

    def on_stop(self):
        # Cleanly close the database connection when the application is closed
        asyncio.run(db.close())

def main():
    return VoiceAssistantApp()

if __name__ == '__main__':
    main().main_loop()