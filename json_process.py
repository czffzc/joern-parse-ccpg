import json

def save_cpg_to_file(cpg_dict, output_file_path):
    """
    将 CPG 的字典对象保存为 JSON 文件
    :param cpg_dict: Joern 生成的 CPG 数据，应该是一个字典对象
    :param output_file_path: 保存的目标文件路径
    """
    try:
        # 将字典对象直接保存为 JSON 文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(cpg_dict, f, ensure_ascii=False, indent=4)

        print(f"CPG successfully saved to {output_file_path}")

    except Exception as e:
        print(f"Error saving CPG to file: {e}")

def save_json_string_to_file(json_string, output_file_path):
    """
    将 Joern 返回的 JSON 字符串直接保存到文件
    :param json_string: Joern 返回的 JSON 格式字符串
    :param output_file_path: 保存的目标文件路径
    """
    try:
        # 保存原始 JSON 字符串到文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(json_string)

        print(f"CPG JSON successfully saved to {output_file_path}")

    except Exception as e:
        print(f"Error saving JSON to file: {e}")
