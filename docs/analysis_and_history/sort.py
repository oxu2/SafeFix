import json
from collections import OrderedDict

# 1. 读取原始 JSON 文件
with open('save/feasible_sademotion.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 对条目按键名前的数字部分排序
#    e.g. key = "016069_sademotion" -> prefix = "016069" -> int(prefix) = 16069
sorted_items = sorted(
    data.items(),
    key=lambda kv: int(kv[0].split('_')[0])
)

# 3. 构造一个有序字典以保持排序
sorted_data = OrderedDict(sorted_items)

# 4. 将排序后的字典写入新文件
with open('save/feasible_sademotion_sorted.json', 'w', encoding='utf-8') as f:
    json.dump(sorted_data, f, indent=4, ensure_ascii=False)

print("✅ Done! Sorted JSON saved to 'save/feasible_sademotion_sorted.json'")