# Voice-Activated Personal Assistant
___
[![Build Voice-Activated Chat System Executable](https://github.com/LoQiseaking69/VoiceChatassist/actions/workflows/build.yml/badge.svg)](https://github.com/LoQiseaking69/VoiceChatassist/actions/workflows/build.yml)
___
![Voice-Activated Personal Assistant](chatv2.png)

## Overview
This project develops a voice-activated personal assistant that uses proven tools and technologies to understand and respond effectively to user commands. Designed for dependable operation across various environments, it delivers clear and direct responses, enhancing user interaction.

## Key Features
- **Offline Speech Recognition**: Utilizes `speech_recognition` and `pyttsx3` for converting spoken language to text and text to spoken language, all processed locally to ensure user privacy.
- **Natural Language Understanding**: Integrates `spacy` with the `en_core_web_trf` model for comprehensive parsing and understanding of text, enabling the assistant to recognize spoken commands and contextualize them accurately.
- **Persistent Storage**: Employs a SQLite database managed through `aiosqlite` for efficient and high-performance asynchronous data operations, storing interaction logs and usage frequencies without performance degradation.
- **Scalable Design**: The system's architecture, while straightforward, is built to accommodate growth and further integration. It includes robust error handling and logging, facilitating easier maintenance and expansion.

## Installation
Set up this personal assistant on your local machine by following these steps:

```bash
# Clone the repository
git clone https://github.com/agreene90/VoiceChatassist
cd voice-activated-assistant

# Make the deploy script executable
chmod +x deploy.sh

# Run the deploy script to set up the environment
./deploy.sh
```

## Usage
Interact with the assistant by sending audio to the `/process/` endpoint through our API. The system processes these requests efficiently, ensuring timely and accurate responses.

## Architecture
The system is structured into several essential components:
- **Speech Module**: Converts spoken language into text and text back into spoken language, operating completely offline.
- **NLP Module**: Analyzes the text to extract meaningful information and understand the context.
- **Database Module**: Manages data interactions and logs efficiently with non-blocking database operations.
- **Utilities**: Supports the system with necessary backend services like logging and error handling.

## Deployment
Follow the provided `deploy.sh` script instructions for easy deployment on any compatible server. This script prepares all necessary environment settings and dependencies.

## License
The project is open-sourced under the MIT License, allowing free modification and distribution of the software while providing protection for the original authors. See the `LICENSE` file for more details.
