# Local DeepSeek Chat Interface

This project provides a simple chat interface for interacting with DeepSeek and Mistral LLMs running locally via Ollama.

## Prerequisites

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull the models you want to use:
```bash
# DeepSeek Models
# Smaller model (1.5B parameters)
ollama pull deepseek-r1:1.5b
# Larger model (7B parameters)
ollama pull deepseek-r1:7b

# Mistral Model
ollama pull mistral:7b
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Chat Interface

1. Start the chat interface:
```bash
python chat.py
```

2. Available commands:
   - Type your messages and press Enter to chat
   - Type 'switch' to switch between models
   - Type 'quit' to exit 