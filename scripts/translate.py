# translate.py (LangChain + ChromaDB version)
import os
import json
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


# ----------------------------------------
# Paths
# ----------------------------------------
projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\"
binFolder = os.path.join(projectRoot, "bin") + "\\"
chromaFolder = os.path.join(binFolder, "chroma_glossary")  # must match previous save
translationFolder = os.path.join(projectRoot, "translation") + "\\"

# ----------------------------------------
# Load config
# ----------------------------------------
with open(projectRoot + "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
with open(projectRoot + "local.json", "r", encoding="utf-8") as f:
    local = json.load(f)

activeClient = config["activeClient"]
clientConfig = config[activeClient]
localConfig = local[activeClient]

# ----------------------------------------
# Embedding + LLM setup
# ----------------------------------------
embedding = OpenAIEmbeddings(
    model=clientConfig["model"],
    api_key=localConfig["key"],
    base_url=clientConfig.get("base_url")
)

chat_llm = ChatOpenAI(
    model=clientConfig["chatModel"],
    api_key=localConfig["key"],
    base_url=clientConfig.get("base_url"),
    temperature=0.3
)

# ----------------------------------------
# Load / build Chroma Vector DB
# ----------------------------------------
def load_or_build_chroma():
    if os.path.exists(chromaFolder):
        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=chromaFolder,
            collection_name="glossary"
        )
        print("✅ Loaded existing Chroma DB.")
    else:
        print("❌ Chroma DB not found. Please run the glossary embedding script first.")
        exit(1)
    return vectordb

vectordb = load_or_build_chroma()
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----------------------------------------
# Helper functions
# ----------------------------------------
def strict_lookup(zh_term: str):
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 5,                        # max rows to return
            "filter": {"zh": zh_term}      # STRICT EXACT MATCH
        }
    )
    return retriever.invoke(zh_term)  # query text can be anything

def find_all_terms(query: str):
    """
    Detect all Chinese glossary terms that appear in the query,
    using strict_lookup to ensure only exact glossary terms are considered.
    Returns a list of detected zh terms.
    """
    # Step 1: Get all zh terms from Chroma metadata
    all_items = vectordb._collection.get(include=["metadatas"])
    all_zh_terms = [md.get("zh") for md in all_items["metadatas"] if md.get("zh")]

    # Step 2: Sort terms by length descending (longest first) to handle overlapping terms
    all_zh_terms.sort(key=len, reverse=True)

    # Step 3: Detect terms in query strictly
    detected = []
    for zh_term in all_zh_terms:
        if zh_term in query and strict_lookup(zh_term):
            # Avoid partial overlap: skip if already part of a longer detected term
            if not any(zh_term in existing for existing in detected):
                detected.append(zh_term)

    return detected

def translate_with_glossary(query: str, detected_terms: list = [], top_k: int = 3):
    # ----------------------------------------------------
    # Step 1: For each term, get strict glossary entries
    # ----------------------------------------------------
    glossary_entries = []
    for term in detected_terms:
        docs = strict_lookup(term)
        for d in docs:
            glossary_entries.append(d.metadata)

    # ----------------------------------------------------
    # Step 2: Build glossary text for LLM
    # ----------------------------------------------------
    glossary_context = ""
    zh_to_en = {}  # mapping for replacement
    for item in glossary_entries:
        zh = item["zh"]
        # en = item.get("selected_text", "")
        en = item.get("abc", "")
        glossary_context += f"- {zh} → {en}\n"
        zh_to_en[zh] = en

    if not glossary_context:
        glossary_context = "(无匹配术语)"

    # ------------------------------
    # translation prompt
    # ------------------------------
    prompt = f"""
你是一个专业的佛学翻译助手。请将下面的中文句子翻译成英文，
并严格参考下列术语表中的译法。

规则：
1. 若句中包含的术语出现在术语表中，必须使用中括号标注。
2. 若句中包含的术语出现在术语表中，必须完全按照术语表翻译。

术语表：
{glossary_context}

需要翻译的句子：
{query}
"""

    resp = chat_llm.invoke([
        {"role": "system", "content": "You are a professional Buddhist translator."},
        {"role": "user", "content": prompt}
    ])
    
    translation = resp.content.strip()
    return translation

# ----------------------------------------
# Main loop
# ----------------------------------------
if __name__ == "__main__":
    print("📘 AI 佛学术语翻译器（LangChain + Chroma 版本）")
    print("输入中文句子（输入 exit 退出）：")

    while True:
        query = input("\n> ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "bye"):
            break

        detected_terms = find_all_terms(query)
        translation = translate_with_glossary(query, detected_terms)
        print("🔹 匹配术语（中文）:", detected_terms)
        print("🔹 翻译结果（英文）:", translation)