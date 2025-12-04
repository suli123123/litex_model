from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import tqdm
from datetime import datetime
import numpy as np
from tools import *

# 配置
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "results/checkpoint-1631"
MAX_RETRIES = 5
OUTPUT_DIR = "./evaluation_logs_retry"

# 加载数据
test_dataset = load_dataset_from_json("dataset/dataset_test_100.json")
test_dataset = test_dataset.map(format_training_example)["train"]
print(f"评估集大小：{len(test_dataset)}")

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()
model.eval()
print("模型加载完成！")

def execute_generation_step(messages, max_new_tokens=512, temperature=0.7):
    """执行单次生成"""
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def generate_code_with_correction(user_input, max_retries=3):
    """带有自我修正循环的生成流程"""
    messages = [{"role": "user", "content": user_input}]
    current_code = ""
    final_error_log = ""
    is_syntax_valid = False
    attempts_used = 0
    
    for attempt in range(max_retries + 1):
        attempts_used = attempt
        current_code = execute_generation_step(messages)
        
        # 语法检查
        success, error_log = verify_litex_syntax(current_code)
        
        if success:
            is_syntax_valid = True
            break
        
        # 如果失败且有剩余重试次数，构造反馈
        if attempt < max_retries:
            messages.append({"role": "assistant", "content": current_code})
            feedback_prompt = (
                f"The Litex code you generated failed to compile.\n"
                f"Error Message:\n{error_log}\n\n"
                f"Please fix the syntax errors based on the error message and output the complete, correct Litex code."
            )
            messages.append({"role": "user", "content": feedback_prompt})
        
        final_error_log = error_log

    return {
        "final_code": current_code,
        "is_syntax_valid": is_syntax_valid,
        "retries_used": attempts_used,
        "error_log": final_error_log
    }

def run_iterative_evaluation(data_subset, num_samples=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if num_samples is not None:
        data_subset = data_subset.select(range(min(num_samples, len(data_subset))))
    
    results = []
    
    print(f"开始评估 {len(data_subset)} 个样本 (Max Retries: {MAX_RETRIES})...")
    
    # 修复点：使用 tqdm.tqdm
    for i, example in tqdm.tqdm(enumerate(data_subset)):
        try:
            # 1. 生成与修正
            gen_result = generate_code_with_correction(example["user_input"], max_retries=MAX_RETRIES)
            
            final_code = gen_result["final_code"]
            is_syntax_valid = gen_result["is_syntax_valid"]
            retries = gen_result["retries_used"]
            
            # 2. 语义检查 (仅当语法通过时)
            semantic_valid = False
            if is_syntax_valid:
                # 使用 tools 中的语义验证函数
                check_result = verify_litex_semantics({
                    "id": example["id"],
                    "nl_problem": example["nl_problem"],
                    "formal_statement": final_code
                })
                semantic_valid = (check_result.get("actual", "").strip().lower() == "yes")
            
            # 实时反馈
            if is_syntax_valid and retries > 0:
                 # 修复点：使用 tqdm.tqdm.write
                 tqdm.tqdm.write(f"✅ Sample {example['id']} fixed after {retries} retries!")

            results.append({
                "id": example["id"],
                "input": example["nl_problem"],
                "final_code": final_code,
                "syntax_valid": is_syntax_valid,
                "semantic_valid": semantic_valid,
                "retries_used": retries,
                "pass_at_1": (is_syntax_valid and retries == 0), # 新增：一次通过
                "fixed_by_retry": (is_syntax_valid and retries > 0),
                "error_log": gen_result["error_log"] if not is_syntax_valid else None
            })

        except Exception as e:
            # 修复点：使用 tqdm.tqdm.write 防止打印错乱
            tqdm.tqdm.write(f"Error processing {example['id']}: {e}")
            continue

    # ================= 计算统计指标 (新增部分) =================
    total = len(results)
    if total == 0:
        print("无结果生成。")
        return

    # 1. 语法相关指标
    syntax_pass_count = sum(1 for r in results if r["syntax_valid"])
    pass_at_1_count = sum(1 for r in results if r["pass_at_1"])
    fixed_count = sum(1 for r in results if r["fixed_by_retry"])
    
    # 2. 语义相关指标 (前提是语法通过)
    semantic_pass_count = sum(1 for r in results if r["semantic_valid"])
    
    # 3. 平均重试次数 (仅计算最终修正成功的样本)
    fixed_samples_retries = [r["retries_used"] for r in results if r["fixed_by_retry"]]
    avg_retries_for_fix = np.mean(fixed_samples_retries) if fixed_samples_retries else 0

    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": total,
        
        # 语法指标
        "syntax_pass_rate": round(syntax_pass_count / total, 4),      # 最终语法通过率
        "pass_at_1_rate": round(pass_at_1_count / total, 4),          # 一次通过率 (Pass@1)
        "retry_success_rate": round(fixed_count / total, 4),          # 重试修复贡献率
        "avg_retries_when_fixed": round(float(avg_retries_for_fix), 2), # 修复样本的平均重试成本
        
        # 语义指标
        "semantic_pass_rate": round(semantic_pass_count / total, 4),  # 最终语义正确率 (Semantic Acc)
        "overall_success_rate": round(semantic_pass_count / total, 4) # 同上 (语法+语义都对才算对)
    }
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"retry_results_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"stats": stats, "details": results}, f, ensure_ascii=False, indent=2)
        
    # 打印最终报表
    print("\n" + "="*50)
    print(f" Evaluation Report ({total} samples)")
    print("="*50)
    print(f" Syntax Pass Rate (Final):   {stats['syntax_pass_rate']:.2%}")
    print(f" Pass@1 (No Retries):        {stats['pass_at_1_rate']:.2%}")
    print(f" Fixed by Self-Correction:   {stats['retry_success_rate']:.2%} (Count: {fixed_count})")
    print(f" Avg Retries (Fixed Cases):  {stats['avg_retries_when_fixed']:.2f}")
    print("-" * 30)
    print(f" Semantic Accuracy:          {stats['semantic_pass_rate']:.2%}")
    print("="*50)
    print(f"Logs saved to: {output_file}")

if __name__ == "__main__":
    run_iterative_evaluation(test_dataset, num_samples=None)