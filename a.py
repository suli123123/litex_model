# download_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
SAVE_PATH = "./models/Qwen2.5-7B-Instruct"  # 本地保存路径


def download_model():
    """下载模型和tokenizer到本地"""
    print(f"开始下载模型: {MODEL_PATH}")
    print("此过程可能需要一些时间，取决于网络速度和模型大小...")

    try:
        # 下载tokenizer
        print("正在下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"✅ Tokenizer已保存到: {SAVE_PATH}")

        # 下载模型
        print("正在下载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.save_pretrained(SAVE_PATH)
        print(f"✅ 模型已保存到: {SAVE_PATH}")

        print("🎉 模型下载完成！")
        print(f"你可以修改训练代码中的 MODEL_PATH 为: '{SAVE_PATH}'")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

    return True


if __name__ == "__main__":
    # 检查是否已安装transformers
    try:
        import transformers
    except ImportError:
        print("❌ 请先安装transformers: pip install transformers")
        exit(1)

    # 检查是否已安装torch
    try:
        import torch
    except ImportError:
        print("❌ 请先安装torch: pip install torch")
        exit(1)

    download_model()