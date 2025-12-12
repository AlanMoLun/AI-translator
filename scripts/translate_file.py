# =====================================================
# 4️⃣ 主程序入口（文件交互版）
# =====================================================
import os
from translate import translate_with_glossary
from translate import find_all_terms

if __name__ == "__main__":
    # 定义目录
    projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\"
    glossariesFolder = os.path.join(projectRoot, "glossaries") + "\\"
    translationFolder = os.path.join(projectRoot, "translation") + "\\"
    input_file = os.path.join(translationFolder, "input.txt")
    output_file = os.path.join(translationFolder, "translated.txt")

    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        exit(1)

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    all_translations = []
    for query in lines:
        detected_terms = find_all_terms(query)
        translation = translate_with_glossary(query, detected_terms)

        # 拼接输出：翻译 + 参考术语
        result = f"原文：{query}\n翻译：{translation}\n参考术语：{detected_terms}\n{'-'*50}"
        all_translations.append(result)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_translations))

    print(f"🎉 翻译完成，结果已写入: {output_file}")
