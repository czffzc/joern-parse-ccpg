import json
import re
from tqdm import tqdm

with open('/home/fzc/dataset/deeprace/deeprace_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

    # 遍历data
for item in tqdm(data):
        # item['func']去除那些以#开头且以数字结尾的行
    item['func'] = [line for line in item['func'] if not re.match(r'# \d+ "<.*>"', line)]
        
    # print(item)
    # 写回到文件
with open('/home/fzc/dataset/deeprace/deeprace_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)