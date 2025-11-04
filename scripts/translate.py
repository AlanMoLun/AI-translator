# translate.py
# ----------------------------------------
# 从 glossary.pkl + glossary.index 加载词库
# 使用向量匹配找到相关术语
# 并用 OpenAI 翻译输入句子，自动参考术语表

import os
import json
import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

# 定义目录
projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\"
binFolder = os.path.join(projectRoot, "bin") + "\\"
glossariesFolder = os.path.join(projectRoot, "glossaries") + "\\"
translationFolder = os.path.join(projectRoot, "translation") + "\\"

# =====================================================
# 1️⃣ 读取配置文件
# =====================================================
with open(projectRoot + "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open(projectRoot + "local.json", "r") as f:
    local = json.load(f)

activeClient = config["activeClient"]
clientConfig = config[activeClient]
localConfig = local[activeClient]

# 推导文件名
pklFile = binFolder + "glossary.pkl"
indexFile = binFolder + "glossary.index"

# 初始化 embedding 客户端
clientParams = {"api_key": localConfig["key"]}
if "base_url" in clientConfig:
    clientParams["base_url"] = clientConfig["base_url"]
if activeClient == "azure" and "api_version" in clientConfig:
    clientParams["api_version"] = clientConfig["api_version"]

client = OpenAI(**clientParams)

# 初始化 chat 客户端
chatClientParams = {"api_key": localConfig["key"]}
if "base_url" in clientConfig:
    chatClientParams["base_url"] = clientConfig["base_url"]
if "api_version" in clientConfig:
    chatClientParams["api_version"] = clientConfig["api_version"]

chat_client = OpenAI(**chatClientParams)

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

# -------------------- 高亮术语使用情况（返回字符串） --------------------
def get_terms_usage_string(translation: str, terms: list) -> str:
    """
    返回参考术语使用情况的字符串。
    支持大小写忽略和简单变形判断。
    """
    lines = ["\n🔸 参考术语："]
    
    if not terms:
        lines.append("  （无匹配术语）")
        return "\n".join(lines)

    translation_lower = translation.lower().replace("\n", " ").strip()

    for t in terms:
        term_trans = t["translation"].lower().strip()
        # 忽略空翻译
        if not term_trans or term_trans in ("-", ""):
            used_flag = "❌ 未使用"
        else:
            # 拆分斜杠形式的多选翻译
            options = [opt.strip() for opt in term_trans.split("/") if opt.strip()]
            used_flag = "❌ 未使用"
            for opt in options:
                if opt in translation_lower:
                    used_flag = "✅ 已使用"
                    break

        lines.append(f"  {t['zh']} → {t['translation']}（{t['source_column']}） {used_flag}  [距离: {t['distance']:.4f}]")

    return "\n".join(lines)


def get_embedding(text: str):
    """生成文本的 embedding 向量"""
    resp = client.embeddings.create(
        model=clientConfig["model"],
        input=text
    )
    return np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)


def find_similar_terms(query: str, top_k: int = 3):
    """
    查找与输入文本最相似的术语。
    优先使用字符串直接匹配，然后再用 embedding 搜索补充。
    """
    matches = []

    # 1. 直接匹配：优先找出句中明确包含的术语
    for _, row in glossary_data.iterrows():
        zh_term = str(row["zh"]).strip()
        if zh_term and zh_term in query:
            matches.append({
                "zh": zh_term,
                "translation": row["selected_text"] if "selected_text" in row else row.get("en", ""),
                "source_column": row.get("source_column", "N/A"),
                "distance": 0.0
            })

    # 2. 向量匹配：补充语义上相近的术语
    try:
        query_emb = get_embedding(query)
        D, I = index.search(query_emb, top_k)
        for dist, idx in zip(D[0], I[0]):
            row = glossary_data.iloc[idx]
            zh_term = str(row["zh"]).strip()
            if zh_term not in [m["zh"] for m in matches]:
                matches.append({
                    "zh": zh_term,
                    "translation": row["selected_text"] if "selected_text" in row else row.get("en", ""),
                    "source_column": row.get("source_column", "N/A"),
                    "distance": float(dist)
                })
    except Exception as e:
        print(f"⚠️ 向量搜索失败: {e}")

    # 3. 按距离排序（直接匹配的 distance=0 优先）
    matches.sort(key=lambda x: x["distance"])
    return matches

def translate_with_glossary(query: str, auto_detect_terms: bool = True):
    """
    结合 glossary 信息的智能翻译。
    在翻译前整合字符串匹配和语义匹配结果。
    """
    detected_terms = []
    
    # =====================================================
    # 🧩 可选：使用 AI 自动检测疑似佛学术语
    # =====================================================
    if auto_detect_terms:
        try:
            term_resp = client.chat.completions.create(
                model=clientConfig["chatModel"],
                messages=[
                    {"role": "system", "content": "你是一个佛学术语识别助手。"},
                    {"role": "user", "content": f"列出以下句子中出现的佛学术语（只列中文术语）：\n{query}"}
                ]
            )
            # 提取检测到的术语（用 split() 自动去除多余空格、换行）
            detected_terms = term_resp.choices[0].message.content.split()
            if detected_terms:
                print(f"🧠 检测到疑似术语: {', '.join(detected_terms)}")
        except Exception as e:
            print(f"⚠️ 检测术语时出错: {e}")
            detected_terms = []
    
    # =====================================================
    # 🔍 查找与输入文本或检测术语相似的词汇
    # =====================================================
    similar_terms = []
    if detected_terms:
        # 对检测到的每个术语单独查找相似项
        for term in detected_terms:
            similar_terms.extend(find_similar_terms(term, top_k=2))
    else:
        # 否则只查整句
        similar_terms = find_similar_terms(query)
    
    # 去重：根据中文术语去重
    seen = set()
    unique_terms = []
    for t in similar_terms:
        if t["zh"] not in seen:
            seen.add(t["zh"])
            unique_terms.append(t)

    # 构建术语提示文本
    if similar_terms:
        glossary_context = "\n".join(
            [f"{t['zh']} → {t['translation']}（{t['source_column']}）" for t in similar_terms]
        )
    else:
        glossary_context = "（无匹配术语）"

    # 构建翻译 prompt
    prompt = f"""
你是一个专业的佛学翻译助手。请将下面的中文句子翻译成英文，
并严格参考下列术语表中的译法。
⚠️ 若句中包含的术语出现在术语表中，必须完全按照术语表翻译。

术语表：
{glossary_context}

需要翻译的句子：
{query}

请输出忠实、自然、专业的英文翻译。
"""

    resp = client.chat.completions.create(
        model=clientConfig["chatModel"],
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

        translation, terms = translate_with_glossary(query, False)

        print("\n🔹 翻译结果：")
        print(translation)

        # print("\n🔸 参考术语：")
        # for t in terms:
        #     print(f"  {t['zh']} → {t['translation']} ({t['source_column']})  [距离: {t['distance']:.4f}]")
        
        # 高亮术语使用情况
        terms_str = get_terms_usage_string(translation, terms)
        print(terms_str)

