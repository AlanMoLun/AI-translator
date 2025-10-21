# 🧩 AI Glossary Translator

基于术语表（Glossary）的智能翻译工具。通过为术语生成向量索引并在翻译时自动检索相似术语，帮助你保持术语的一致性与专业性。支持 OpenAI、Azure、DeepSeek 等兼容 OpenAI SDK 的客户端。

---

## 📁 项目目录结构

```
AI_Glossary_Translator/
│
├─ bin/                      # 自动生成的二进制/索引文件
│   ├─ glossary.index        # FAISS 向量索引
│   └─ glossary.pkl          # 词库 + 嵌入向量
│
├─ glossaries/               # 术语表 CSV 目录
│   ├─ demo.csv
│   ├─ glossary.csv
│   └─ glossary_V2.csv
│
├─ scripts/                  # 脚本
│   ├─ create_embeddings.py  # 生成嵌入与索引
│   └─ translate.py  # 使用词库辅助翻译（交互式）
│   └─ translate_file.py  # 使用词库辅助翻译（文件式）
│
├─ translation/
│   ├─ input.txt             # 待翻译文本（可选）
│   └─ translated.txt        # 输出结果（若脚本写入输出）
│
├─ config.json               # 主配置
├─ local.json                # 本地密钥/私有配置（不应提交到版本库）
├─ requirements.txt          # 依赖
└─ README.md                 # 项目说明（本文档）
```

---

## 📦 各目录与文件说明

- bin/
  - glossary.pkl：序列化后的词库数据（含所用文本、来源列、嵌入向量）
  - glossary.index：FAISS 生成的向量索引，用于相似术语检索
  - 提示：当词库 CSV 变更时需要重新生成这两类文件

- glossaries/
  - 存放原始术语表 CSV
  - 推荐字段（至少包含以下关键列，程序将按优先级选择用于嵌入/翻译参照的文本）：
    - zh：中文术语
    - en：英文翻译（优先）
    - en_v：英文备选
    - tibetan：藏文（可选）
    - pali_sanskrit：梵/巴利文（备选）
    - note/rendering/tradition/source：注释/音译/传统译法/来源（可选）

- scripts/
  - create_embeddings.py：读取 glossaries 下的 CSV，生成 embedding 与索引，输出到 bin/
  - translate.py：交互式翻译，自动检索相似术语并在提示中给出参照
  - translate_file.py：文件翻译

- translation/
  - 用于放置批量翻译的输入与输出文件（当前脚本以交互为主，可按需扩展为文件模式）

- 配置文件
  - config.json：全局配置（激活客户端、词库路径、模型等）
  - local.json：本地私密配置（API Key 等），建议在 .gitignore 中忽略

---

## ⚙️ 配置示例

config.json（示例）：

```json
{
  "activeClient": "openai",
  "glossary": "glossaries/glossary.csv",
  "openai": {
    "base_url": "https://api.openai.com/v1",
    "model": "text-embedding-3-small"
  },
  "azure": {
    "base_url": "https://xxx.openai.azure.com",
    "api_version": "2024-06-01-preview",
    "model": "gpt-4o-mini"
  },
  "deepseek": {
    "base_url": "https://api.deepseek.com",
    "model": "text-embedding-3-small"
  }
}
```

local.json（示例，仅存放私密 Key）：

```json
{
  "openai": { "key": "你的-OpenAI-API-Key" },
  "azure":  { "key": "你的-Azure-OpenAI-API-Key" },
  "deepseek": { "key": "你的-DeepSeek-API-Key" }
}
```

注意：activeClient 需与 config.json 中的同名字段对应（如 "openai"、"azure"、"deepseek"）。脚本会读取 config[activeClient] 与 local[activeClient]。

---

## 🚀 快速开始

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 准备词库

- 将 CSV 放到 glossaries/ 目录（默认使用 config.json 中的路径）
- 至少包含列：zh、en、en_v、pali_sanskrit（脚本会按 en → en_v → pali_sanskrit → zh 的优先级选择文本）

3) 生成嵌入与索引

```bash
python scripts/create_embeddings.py
```

运行成功后，会在 bin/ 下生成 glossary.pkl 与 glossary.index。

4) 交互式翻译

```bash
python scripts/translate.py
```

- 启动后直接输入中文句子回车
- 程序会展示英文翻译以及参考到的相似术语（来源列、相似度等）
- 输入 exit/quit/bye 退出

5) 文件翻译

```bash
python scripts/translate_file.py
```

- 将要翻译文件命名为input.txt并存放到translation目录下
- 程序会展示英文翻译以及参考到的相似术语（来源列、相似度等）
- 翻译完成，结果写入: .\translation\translated.txt

---

## 🧠 工作原理概览

- 通过 create_embeddings.py：
  - 读取术语表，按优先级选择用于嵌入的文本（en > en_v > pali_sanskrit > zh）
  - 使用所选客户端的 embedding 模型生成向量
  - 构建 FAISS 索引并落盘（bin/glossary.index）
  - 将包含术语、所用文本、来源列、向量等信息的 DataFrame 序列化为 bin/glossary.pkl

- 通过 translate.py：
  - 对用户输入句子生成 embedding
  - 在 FAISS 索引中检索若干最相似术语，拼装为“术语上下文”
  - 调用 Chat 模型进行翻译，要求参考术语上下文，保证用词一致性

---

## ✅ 小贴士

- 词库变更后，需重新运行 create_embeddings.py 以更新 bin/ 下索引与数据
- 切换 API 供应商：修改 config.json 的 activeClient，并在 local.json 中提供对应 key
- 对于大型词库，可考虑使用带 GPU 的 FAISS 版本以加速检索
- 建议将 local.json、bin/ 等加入 .gitignore，避免提交私密/大文件

---

## ❓常见问题

- 报错提示找不到 local.json：请在项目根目录创建 local.json，并填入对应 activeClient 的 key
- 索引加载失败：确认已运行过 create_embeddings.py，且 bin/glossary.index 存在
- 术语列缺失：确保 CSV 至少包含 zh、en、en_v、pali_sanskrit 中的若干列（脚本会检查必需列）

---

## 📄 许可证
MIT