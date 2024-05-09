import asyncio
import toga
from toga.style import Pack
from toga.style.pack import COLUMN
from cryptography.fernet import Fernet

# Import custom modules
from .database_interaction import ChatDatabase
from .ML import MLModel
from .speech_recognition import recognizer, microphone  # Assuming specific imports if needed
from .utilities import CustomLogger, EncryptedFileHandler
from .nlp_processing import listen_and_respond, analyze_text  # Include your NLP processing functions

# Generate or retrieve an existing secure encryption key
encryption_key = Fernet.generate_key()  # This should be securely stored and retrieved in a production environment

# Setup custom logging with encryption
logger = CustomLogger('app.log', max_bytes=1000000, backup_count=5, encryption_key=encryption_key)

# Initialize the ML Model with advanced NLP features
ml_model = MLModel(logger)

# Initialize the database connection
db = ChatDatabase('chat_memory.db')
asyncio.run(db.init_db())  # Ensure the database is initialized

class VoiceAssistantApp(toga.App):
    def startup(self):
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

    async def process_audio(self):
        # Capture and process speech using the integrated NLP processing function
        text = listen_and_respond()  # This function now captures and processes speech
        if text:
            # Analysis and response generation based on the text processed
            analysis_results = analyze_text(text)
            entities, sentiment, summary = analysis_results
            logger.log_info(f"Processed interaction: Text: '{text}', Entities: {entities}, Sentiment: {sentiment}, Summary: {summary}")
            await db.update_response_frequency(summary)
            response = f"Entities: {entities}\nSentiment: {sentiment}\nSummary: {summary}"
            self.response_label.text = response  # Display the analysis results in the UI

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