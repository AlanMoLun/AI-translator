# =====================================================
# 4️⃣ 主程序入口（文件交互版）
# =====================================================
import os
from translate import translate_with_glossary
from translate import get_terms_usage_string

if __name__ == "__main__":
    # 定义目录
    projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\"
    binFolder = os.path.join(projectRoot, "bin") + "\\"
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
        translation, terms = translate_with_glossary(query)
        # 拼接输出：翻译 + 参考术语
        terms_str = "\n".join(
            [f"{t['zh']} → {t['translation']} ({t['source_column']})  [距离: {t['distance']:.4f}]"
             for t in terms]
        )
        # result = f"原文：{query}\n翻译：{translation}\n参考术语：\n{terms_str}\n{'-'*50}"
        terms_str = get_terms_usage_string(translation, terms)
        result = f"原文：{query}\n翻译：{translation}\n参考术语：\n{terms_str}\n{'-'*50}"
        all_translations.append(result)
        print(f"✅ 已翻译: {query}")

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_translations))

    print(f"\n🎉 翻译完成，结果已写入: {output_file}")
