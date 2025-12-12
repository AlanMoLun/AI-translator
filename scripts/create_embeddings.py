import os
import pandas as pd
import json
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

# =========================
# 目录设置
# =========================
projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\"
binFolder = os.path.join(projectRoot, "bin") + "\\"
glossariesFolder = os.path.join(projectRoot, "glossaries") + "\\"

# =========================
# 读取配置
# =========================
with open(projectRoot + "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open(projectRoot + "local.json", "r", encoding="utf-8") as f:
    local = json.load(f)

activeClient = config["activeClient"]
glossaryFile = glossariesFolder + config["glossary"]
clientConfig = config[activeClient]
localConfig = local[activeClient]

print(f"使用客户端: {activeClient}")
print(f"使用词库文件: {glossaryFile}")

# =========================
# 初始化 OpenAI Embeddings
# =========================
clientParams = {"openai_api_key": localConfig["key"]}
embeddings_model = clientConfig.get("model", "text-embedding-3-large")
embeddings = OpenAIEmbeddings(model=embeddings_model, **clientParams)

# =========================
# 读取词库
# =========================
try:
    df = pd.read_csv(glossaryFile)
    print(f"成功读取词库文件: {glossaryFile}, 共 {len(df)} 行")
except Exception as e:
    print(f"读取词库文件出错: {e}")
    exit(1)

required_columns = ["zh", "en", "en_v", "pali_sanskrit"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"错误: 缺少列 {missing_columns}")
    exit(1)

# =========================
# 生成 embedding 数据
# =========================
texts = []
metadatas = []

for idx, row in df.iterrows():
    # 优先级选择文本
    if pd.notna(row["en"]) and str(row["en"]).strip():
        text = str(row["en"]).strip()
        source = "en"
    elif pd.notna(row["en_v"]) and str(row["en_v"]).strip():
        text = str(row["en_v"]).strip()
        source = "en_v"
    elif pd.notna(row["pali_sanskrit"]) and str(row["pali_sanskrit"]).strip():
        text = str(row["pali_sanskrit"]).strip()
        source = "pali_sanskrit"
    else:
        text = str(row["zh"]).strip()
        source = "zh"

    texts.append(text)
    metadatas.append({
        "zh": row["zh"],
        "selected_text": text,
        "source_column": source
    })

print(f"准备生成 {len(texts)} 个 embedding")

# =========================
# 保存到 Chroma
# =========================
persist_directory = os.path.join(binFolder, "chroma_glossary")
chroma_db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory=persist_directory,
    collection_name="glossary",
    client_settings=Settings(anonymized_telemetry=False)
)

print(f"✅ Chroma 词库保存完成，目录: {persist_directory}")

# =========================
# 输出统计
# =========================
source_counts = pd.Series([m["source_column"] for m in metadatas]).value_counts()
print("\n文本来源统计:")
for source, count in source_counts.items():
    print(f"  {source}: {count} 个术语 ({count/len(metadatas)*100:.1f}%)")
