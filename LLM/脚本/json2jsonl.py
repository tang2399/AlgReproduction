import json


# 问答式json文件转化为jsonl文件，用于做大模型微调的数据集
def convert_json_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 读取JSON数组

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # 构造messages结构
            converted = {
                "messages": [
                    {"role": "user", "content": item["Question"]},
                    {"role": "assistant", "content": item["Response"]}
                ]
            }
            f.write(json.dumps(converted, ensure_ascii=False) + '\n')


convert_json_to_jsonl('input.json', 'output.jsonl')
