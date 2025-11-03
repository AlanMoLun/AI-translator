import os
from openai import OpenAI
import pandas as pd
import faiss
import numpy as np
import json

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

# 根据配置选择客户端和glossary文件
activeClient = config["activeClient"]
glossaryFile = glossariesFolder + config["glossary"]
clientConfig = config[activeClient]
localConfig = local[activeClient]

print(f"使用客户端: {activeClient}")
print(f"使用词库文件: {glossaryFile}")

# 设置客户端参数
clientParams = {
    "api_key": localConfig["key"]
}

# 为不同客户端设置特定的基础URL
if "base_url" in clientConfig:
    clientParams["base_url"] = clientConfig["base_url"]

# 为 Azure 添加 API 版本
if activeClient == "azure" and "api_version" in clientConfig:
    clientParams["api_version"] = clientConfig["api_version"]

print(f"客户端参数: {clientParams}")

# 初始化客户端
client = OpenAI(**clientParams)

# 读取词库
try:
    df = pd.read_csv(glossaryFile)
    print(f"成功读取词库文件: {glossaryFile}")
    print(f"词库包含 {len(df)} 行")
except Exception as e:
    print(f"读取词库文件 {glossaryFile} 时出错: {e}")
    exit(1)

# 检查必需的列是否存在
required_columns = ["zh", "en", "en_v", "pali_sanskrit"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"错误: 词库文件缺少以下列: {missing_columns}")
    exit(1)

# 生成 embedding
embeddings = []
selected_texts = []  # 记录实际使用的文本
source_columns = []  # 记录文本来源的列名

for idx, row in df.iterrows():
    # 根据优先级选择文本
    text_to_embed = None
    source_column = None
    
    # 优先级: en -> en_v -> pali_sanskrit -> zh
    if pd.notna(row["en"]) and str(row["en"]).strip():
        text_to_embed = str(row["en"]).strip()
        source_column = "en"
    elif pd.notna(row["en_v"]) and str(row["en_v"]).strip():
        text_to_embed = str(row["en_v"]).strip()
        source_column = "en_v"
    elif pd.notna(row["pali_sanskrit"]) and str(row["pali_sanskrit"]).strip():
        text_to_embed = str(row["pali_sanskrit"]).strip()
        source_column = "pali_sanskrit"
    else:
        # 如果所有优先级列都为空，则使用 zh 作为备选
        text_to_embed = str(row["zh"]).strip()
        source_column = "zh"
    
    try:
        resp = client.embeddings.create(
            model=clientConfig["model"],
            input=text_to_embed
        )
        embeddings.append(resp.data[0].embedding)
        selected_texts.append(text_to_embed)
        source_columns.append(source_column)
        print(f"成功生成术语 '{row['zh']}' 的嵌入向量 (来源: {source_column})")
    except Exception as e:
        print(f"生成术语 '{row['zh']}' 的嵌入向量时出错: {e}")
        # 添加一个零向量作为占位符
        embedding_dim = clientConfig.get("embedding_dim", 1024)
        embeddings.append([0] * embedding_dim)
        selected_texts.append("")
        source_columns.append("error")

# 保存到 DataFrame
df["embedding"] = embeddings
df["selected_text"] = selected_texts  # 添加一列显示实际使用的文本
df["source_column"] = source_columns  # 添加一列显示文本来源的列名

# 保存处理后的数据
output_pkl = binFolder + "glossary.pkl"
output_index = binFolder + "glossary.index"
df.to_pickle(output_pkl)

# 建立 FAISS 索引
matrix = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(matrix.shape[1])
index.add(matrix)

faiss.write_index(index, output_index)

print(f"✅ 使用 {activeClient} 生成词库 embedding 并保存成功。")
print(f"使用的模型: {clientConfig['model']}")
print(f"嵌入维度: {matrix.shape[1]}")
print(f"处理了 {len(embeddings)} 个术语")
print(f"输出文件: {output_pkl} 和 {output_index}")

# 统计使用的文本来源
source_counts = pd.Series(source_columns).value_counts()

print("\n文本来源统计:")
for source, count in source_counts.items():
    percentage = (count / len(source_columns)) * 100
    print(f"  {source}: {count} 个术语 ({percentage:.1f}%)")