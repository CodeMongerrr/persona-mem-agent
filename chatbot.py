from langchain_community.llms import Ollama
from memory import store_memory, recall_memories

llm = Ollama(model="llama3.2")

SYSTEM_PROMPT = """
You are a personal growth and habit tracking assistant.
You remember the user's habits, sleep patterns, goals, and struggles.
Use past memories to give personalized advice.
"""

def chat(user_input: str):
    memories = recall_memories(user_input)

    memory_context = "\n".join(memories)

    prompt = f"""
    {SYSTEM_PROMPT}

    Relevant past memories:
    {memory_context}

    User: {user_input}
    Assistant:
    """

    response = llm.invoke(prompt)

    # Store important memories automatically
    if "I" in user_input:
        store_memory(user_input)

    return response


if __name__ == "__main__":
    print("AdityaOS is online. Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        reply = chat(user_input)
        print(f"AI: {reply}\n")