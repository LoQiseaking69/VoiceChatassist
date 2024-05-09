#!/bin/bash

# Define the script's exit on error
set -e

echo "Starting deployment of the voice-activated personal assistant."

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

# Install required Python packages locally
echo "Installing required Python packages..."
pip install toga spacy nltk pyttsx3 cryptography SpeechRecognition

# Download and setup SpaCy models
echo "Downloading and setting up SpaCy models..."
python -m spacy download en_core_web_trf

# Install NLTK data
echo "Installing NLTK data..."
python -m nltk.downloader punkt

# Ensure the database is initialized properly with error handling
echo "Initializing the database..."
python -c 'import asyncio; from database_interaction import ChatDatabase; db = ChatDatabase("chat_memory.db"); asyncio.run(db.init_db())' || echo "Failed to initialize the database. Check the logs for details."

# The Toga app runs as a standalone GUI application
echo "Launching the application..."
python -m main

echo "Deployment complete."

# Deactivate the virtual environment
deactivate

# Script exit point
echo "Script completed successfully."