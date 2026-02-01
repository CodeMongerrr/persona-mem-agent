from datetime import datetime
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import re
PERSIST_DIR = "./memory_db"

embeddings = OllamaEmbeddings(model="llama3.2")

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

def should_store_memory(llm, user_input: str) -> bool:
    prompt = f"""
Decide if this message contains ANY personal information useful for a personal growth assistant.

This includes:
- goals or plans
- habits or routines
- emotional states
- sleep or health info
- productivity patterns
- upcoming events

Reply ONLY with YES or NO.

Message: "{user_input}"
"""
    decision = llm.invoke(prompt).strip().upper()
    print("🧪 Memory filter decision:", decision)
    return "YES" in decision



def extract_memory(llm, user_input: str):
    prompt = f"""
Extract structured personal memory.

Return ONLY valid JSON. No explanation.

Format:
{{
  "category": "goal | habit | preference | struggle | schedule | health | other",
  "summary": "short memory summary"
}}

Message: "{user_input}"
"""
    response = llm.invoke(prompt)

    try:
        json_str = re.search(r"\{.*\}", response, re.DOTALL)
        if json_str:
            data = json.loads(json_str.group())
            return data
        else:
            print("❌ No JSON found in response")
            print("Raw model output:", response)
    except Exception as e:
        print("❌ JSON Parse Error:", e)
        print("Raw model output:", response)
        return None


def store_memory(llm, text: str):
    if not should_store_memory(llm, text):
        print("⛔ Memory rejected")
        return

    data = extract_memory(llm, text)
    if not data:
        print("⚠️ Memory extraction failed")
        return

    memory_text = f"[{data['category'].upper()}] {data['summary']}"

    metadata = {
        "type": data["category"],
        "timestamp": datetime.now().isoformat()
    }

    print("\n🧠 Storing Memory")
    print("Stored as:", memory_text)
    print("Metadata:", metadata)

    vectorstore.add_texts([memory_text], metadatas=[metadata])

def recall_memories(query: str, k: int = 5, memory_type: str | None = None):
    if memory_type:
        results = vectorstore.similarity_search(
            query, k=k, filter={"type": memory_type}
        )
    else:
        results = vectorstore.similarity_search(query, k=k)

    return [doc.page_content for doc in results]

def view_all_memories(limit=20):
    data = vectorstore.get(include=["documents", "metadatas"])
    print("Total memories:", len(data["documents"]))
    for i, (doc, meta) in enumerate(zip(data["documents"][:limit], data["metadatas"][:limit]), 1):
        print(f"\nMemory #{i}")
        print("Text:", doc)
        print("Metadata:", meta)