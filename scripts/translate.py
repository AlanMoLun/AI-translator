# translate.py
# ----------------------------------------
# 从 glossary.pkl + glossary.index 加载词库
# 使用向量匹配找到相关术语
# 并用 OpenAI 翻译输入句子，自动参考术语表

import os
import json
import pickle
import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

# 定义目录
binFolder = ".\\bin\\"
glossariesFolder = ".\\glossaries\\"
translationFolder = ".\\translation\\"

# =====================================================
# 1️⃣ 读取配置文件
# =====================================================
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open("local.json", "r") as f:
    local = json.load(f)

activeClient = config["activeClient"]
glossaryFile = glossariesFolder + config["glossary"]
clientConfig = config[activeClient]
localConfig = local[activeClient]

# 推导文件名
pklFile = binFolder + "glossary.pkl"
indexFile = binFolder + "glossary.index"

# 初始化 OpenAI 客户端
clientParams = {"api_key": localConfig["key"]}
if "base_url" in clientConfig:
    clientParams["base_url"] = clientConfig["base_url"]
if activeClient == "azure" and "api_version" in clientConfig:
    clientParams["api_version"] = clientConfig["api_version"]

client = OpenAI(**clientParams)

# =====================================================
# 2️⃣ 加载词库和向量索引
# =====================================================
try:
    glossary_data = pd.read_pickle(pklFile)
    print(f"✅ 成功加载词库文件: {pklFile}")
    print(f"词库共有 {len(glossary_data)} 条记录")
except Exception as e:
    print(f"❌ 加载 {pklFile} 失败: {e}")
    exit(1)

try:
    index = faiss.read_index(indexFile)
    print(f"✅ 成功加载索引文件: {indexFile}")
except Exception as e:
    print(f"❌ 加载索引文件 {indexFile} 失败: {e}")
    exit(1)

# =====================================================
# 3️⃣ 定义辅助函数
# =====================================================

def get_embedding(text: str):
    """生成文本的 embedding 向量"""
    resp = client.embeddings.create(
        model=clientConfig["model"],
        input=text
    )
    return np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)


def find_similar_terms(query: str, top_k: int = 3):
    """查找与输入文本最相似的术语"""
    query_emb = get_embedding(query)
    D, I = index.search(query_emb, top_k)
    results = []

    for dist, idx in zip(D[0], I[0]):
        term_row = glossary_data.iloc[idx]
        results.append({
            "zh": term_row["zh"],
            "translation": term_row["selected_text"] if "selected_text" in term_row else term_row.get("en", ""),
            "source_column": term_row.get("source_column", "N/A"),
            "distance": float(dist)
        })
    return results


def translate_with_glossary(query: str):
    """结合 glossary 信息的智能翻译"""
    similar_terms = find_similar_terms(query)
    glossary_context = "\n".join(
        [f"{t['zh']} → {t['translation']} ({t['source_column']})" for t in similar_terms]
    )

    prompt = f"""
你是一个专业的佛学翻译助手。请将下面的中文翻译成英文，
并严格参考下列术语表进行术语一致性翻译。

术语表：
{glossary_context}

需要翻译的句子：
{query}

请输出高质量英文翻译。
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional Buddhist translator."},
            {"role": "user", "content": prompt}
        ]
    )

    translation = resp.choices[0].message.content.strip()
    return translation, similar_terms


# =====================================================
# 4️⃣ 主程序入口
# =====================================================
if __name__ == "__main__":
    print("📘 AI 佛学术语翻译器")
    print("输入中文句子（输入 exit 退出）：")

    while True:
        query = input("\n> ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "bye"):
            break

        translation, terms = translate_with_glossary(query)

        print("\n🔹 翻译结果：")
        print(translation)

        print("\n🔸 参考术语：")
        for t in terms:
            print(f"  {t['zh']} → {t['translation']} ({t['source_column']})  [距离: {t['distance']:.4f}]")
