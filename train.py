from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch
from tools import *

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

# 加载并预处理数据
train_dataset = load_dataset_from_json("dataset/train_litex.json")
test_dataset = load_dataset_from_json("dataset/test_litex.json")

train_dataset = train_dataset.map(format_training_example)["train"]
test_dataset = test_dataset.map(format_training_example)["train"]

print(f"训练集大小：{len(train_dataset)}")
print(f"评估集大小：{len(test_dataset)}")

# 模型与 Tokenizer 加载
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# LoRA 配置
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_steps=250,
    learning_rate=1e-4,
    weight_decay=0.01,
    fp16=True,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps", 
    save_steps=250,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
    report_to=None,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    args=training_args,
)

# 初始评估与训练
print("进行初始评估...")
trainer.evaluate()

trainer.train()

print("\n进行最终评估...")
final_metrics = trainer.evaluate()
print("最终评估结果:", final_metrics)