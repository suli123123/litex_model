litex/

├── train.py                   # 模型微调脚本
├── out.py                     # 带自我修正的评估脚本
├── run.sh                     # 启动训练的 Shell 脚本
├── tools.py                   # 工具函数（数据处理、代码验证等）
│
├── dataset/
│   ├── train_litex.json       # 训练数据集
│   ├── test_litex.json        # 测试数据集
│   └── dataset_test_100.json  # 用于自我修正评估的100个样本
│
├── results/
│   ├── checkpoint-*/          # 训练过程中保存的模型权重
│   └── runs/                    # TensorBoard 日志
│
└── evaluation_logs_retry/
    └── retry_results_*.json   
    

```
环境配置
1.  cuda和torch环境:
    PyTorch  2.8.0
    Python  3.12(ubuntu22.04)
    CUDA  12.8

2.  包依赖:
    项目依赖 PyTorch, Transformers, PEFT, TRL, Accelerate 等库。

    pip install torch transformers peft trl accelerate datasets
```



### 1. 训练模型

通过运行 `run.sh` 脚本来启动模型微调。该脚本会使用 `accelerate` 来执行 `train.py`。

```bash
bash run.sh
```

网络问题，可以切换源
```
export HF_ENDPOINT=https://hf-mirror.com
```

训练脚本会：
*   从 `dataset/train_litex.json` 加载训练数据。
*   使用 LoRA 配置对模型进行微调。
*   定期在 `dataset/test_litex.json` 上进行评估。
*   将最佳模型检查点保存在 `results/` 目录下。

### 2. 评估与推理

运行 `out.py` 脚本来对模型进行评估，运行前先运行`train.py`下载qwen7B模型。

```bash
python out.py
```

评估脚本会：
*   加载指定的基础模型和 `results/` 目录下的 LoRA 检查点。
*   在 `dataset/dataset_test_100.json` 数据集上进行推理，可以选择测试集。
*   对生成的每个代码片段：
    1.  执行语法检查。
    2.  如果语法错误，将错误信息反馈给模型并要求其修正（max_retry = 5）。
    3.  如果语法正确，继续执行语义检查（阿里千问api）。
*   将详细的评估结果（包括代码、语法/语义有效性、重试次数等）保存在 `evaluation_logs_retry/` 目录下的一个 JSON 文件中。

### 3. 数据集来源

litex官方网站，huggingface：https://huggingface.co/litexlang`


## 模型配置

*   **基础模型**: `Qwen/Qwen2.5-7B-Instruct`
*   **微调技术**: LoRA
*   **LoRA 配置 (`train.py`)**:
    *   `r`: 32
    *   `lora_alpha`: 64
