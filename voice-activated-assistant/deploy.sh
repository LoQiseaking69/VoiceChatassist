#!/bin/bash

# Exit the script if any command fails
set -e

echo "Starting deployment of the Voice Activated Chat System..."

# Create and activate a Python virtual environment
echo "Creating and activating a virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Set environment variables
echo "Setting up environment variables..."
export DATABASE_PATH="chat_memory.db"
export MODEL_PATH="models/"
export ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")  # Generating a new encryption key

# Upgrade pip to its latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install required Python packages
echo "Installing required Python packages..."
pip install toga==0.3.0.dev26 \
             cryptography==3.4 \
             spacy==3.0 \
             nltk==3.6 \
             pyttsx3==2.90 \
             SpeechRecognition==3.8 \
             transformers==4.5 \
             aiosqlite==0.17.0 \
             huggingface_hub==0.1.0 \
             beautifulsoup4==4.9.3 \
             requests==2.25 \
             python-dateutil==2.8.1

# Download and setup NLP models
echo "Downloading and setting up SpaCy models..."
python -m spacy download en_core_web_trf

# Install NLTK data
echo "Installing NLTK data..."
python -m nltk.downloader punkt

# Ensure the database is initialized properly with error handling
echo "Initializing the database..."
python -c 'import asyncio; from voice_activated_assistant.database_interaction import ChatDatabase; db = ChatDatabase("${DATABASE_PATH}"); asyncio.run(db.init_db())' || echo "Failed to initialize the database. Check the logs for details."

# Launch the application
echo "Launching the application..."
python -m voice_activated_assistant.main

echo "Deployment complete. You can now use your voice-activated personal assistant."

# Deactivate the virtual environment
deactivate

echo "Script completed successfully."