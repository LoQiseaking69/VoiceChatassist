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
# Generate a new encryption key and export it
export ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Upgrade pip to its latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install required Python packages
echo "Installing required Python packages..."
pip install -r requirements.txt

# Download and setup NLP models
echo "Downloading and setting up SpaCy models..."
python -m spacy download en_core_web_trf

# Install NLTK data
echo "Installing NLTK data..."
python -m nltk.downloader punkt

# Ensure the database is initialized properly with error handling
echo "Initializing the database..."
if python -c 'import asyncio; from database_interaction import ChatDatabase; db = ChatDatabase("${DATABASE_PATH}"); asyncio.run(db.init_db())'; then
    echo "Database initialized successfully."
else
    echo "Failed to initialize the database. Check the logs for details."
    exit 1
fi

# Launch the application
echo "Launching the application..."
python -m voice_activated_assistant.main

echo "Deployment complete. You can now use your voice-activated personal assistant."

# Deactivate the virtual environment
deactivate

echo "Script completed successfully."