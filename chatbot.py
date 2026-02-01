from langchain_community.llms import Ollama
from memory import store_memory, recall_memories
from collections import deque
import json
from datetime import datetime
llm = Ollama(model="llama3.2")

SYSTEM_PROMPT = """
You are a personal growth and habit tracking assistant.
You remember the user's habits, sleep patterns, goals, and struggles.
You are supportive but honest.
You track patterns over time and gently point out inconsistencies.
You celebrate progress.
"""

# Short-term memory buffer
chat_history = deque(maxlen=8)
MEMORY_FILTER_PROMPT = """
Decide if this message contains a long-term personal fact, habit, goal, or preference.
Only answer YES or NO.

Message: "{user_input}"
"""

MEMORY_EXTRACTION_PROMPT = """
Extract structured personal memory from this message.

Return JSON with:
- category: one of [goal, habit, preference, struggle, schedule, health, other]
- summary: short memory to store

Message: "{user_input}"
JSON:
"""


def chat(user_input: str):
    # Long-term memory recall
    memories = recall_memories(user_input)
    memory_context = "\n".join(f"- {m}" for m in memories)

    # Short-term memory
    recent_context = "\n".join(chat_history)

    prompt = f"""
{SYSTEM_PROMPT}

Long-term memories about the user:
{memory_context}

Recent conversation:
{recent_context}

User: {user_input}
Assistant:
"""

    response = llm.invoke(prompt)

    # Store meaningful memories
    store_memory(llm, user_input)

    # Update short-term memory
    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Assistant: {response}")

    return response


if __name__ == "__main__":
    print("AdityaOS is online. Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        reply = chat(user_input)
        print(f"AI: {reply}\n")