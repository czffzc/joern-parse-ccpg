# 去掉源代码中的注释
import os
import re

def remove_comments(source_code):
    source_code = re.sub(r'/\*[\s\S]*?\*/', '', source_code)
    source_code = re.sub(r'//.*?\n', '\n', source_code)
    source_code = re.sub(r'//.*?$', '', source_code)
    
    return source_code

def process_directory(directory_path):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.c'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                cleaned_content = remove_comments(content)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(cleaned_content)
                    
                print(f"已处理文件: {filename}")
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 指定要处理的目录路径
    directory = "/home/kevin/joern-parse/joern-export-demo/c_code/"  # 请替换为实际路径
    
    if os.path.exists(directory):
        process_directory(directory)
        print("所有文件处理完成")
    else:
        print("指定的目录不存在")