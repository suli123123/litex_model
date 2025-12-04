import json

def rename_json_keys(input_path, output_path):
    try:
        # 1. 读取 JSON 文件
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 遍历列表并修改字段
        # 定义旧键名和新键名的映射关系
        key_mapping = {
            "title": "id",
            "description": "nl_problem",
            "full litex": "formal_statement"
        }

        for item in data:
            for old_key, new_key in key_mapping.items():
                # 检查该条目中是否存在旧键
                if old_key in item:
                    # item.pop(old_key) 会删除旧键并返回其值，将其赋值给新键
                    item[new_key] = item.pop(old_key)

        # 3. 将修改后的数据保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文能正常显示，indent=4用于美化格式
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"成功！文件已修改并保存为: {output_path}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
    except Exception as e:
        print(f"发生错误: {e}")

# 执行函数
if __name__ == "__main__":
    # 输入文件名
    input_file = 'train_litex.json'
    # 输出文件名（建议保存为新文件以防覆盖原数据，确认无误后可改回 a.json）
    output_file = 'train_litex_1.json' 
    
    rename_json_keys(input_file, output_file)