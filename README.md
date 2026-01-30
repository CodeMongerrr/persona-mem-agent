# JoshiOS: Personal Growth & Habit Tracking Chatbot

JoshiOS is a simple AI-powered assistant for personal growth, habit tracking, and self-improvement. It remembers your habits, goals, and struggles, and uses past conversations to give personalized advice.

## Features
- Remembers your habits, sleep patterns, goals, and struggles
- Uses memory to provide tailored advice
- Stores and recalls important user inputs
- Runs locally using Ollama and ChromaDB

## Requirements
- Python 3.8+
- [Ollama](https://ollama.com/) (with `llama3.2` model)
- [langchain_community](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB](https://docs.trychroma.com/)

Install dependencies:
```bash
pip install langchain-community chromadb
```

## Usage
1. Start Ollama and ensure the `llama3.2` model is available.
2. Run the chatbot:
   ```bash
   python chatbot.py
   ```
3. Chat with the assistant. Type 'exit' to quit.

## Memory
- User inputs are stored in a local ChromaDB database (`memory_db/`).
- The assistant recalls relevant memories to personalize responses.

## File Structure
- `chatbot.py` — Main chatbot logic
- `memory.py` — Memory storage and retrieval
- `memory_db/` — Persistent memory database

---
**Author:** Aditya Roshan Joshi
