import pylitex
from datasets import load_dataset, Dataset, DatasetDict
import json
import re
import subprocess
from openai import OpenAI, APIStatusError, APIError
import tqdm

# 建议在实际使用中放入环境变量
API_KEY = "sk-cb0af71536e046e4b9d8cf9f5d9660a7"

def verify_litex_syntax(message: str) -> tuple[bool, str]:
    """
    运行 Litex 代码并验证语法。
    返回: (is_success, error_log)
    """
    # 假设 pylitex.run 返回字典
    result = pylitex.run(message)
    
    # 优先提取 log，其次 message
    error_log = result.get("log", "") or result.get("message", "") or ""
    
    # 补充默认错误信息
    if not result["success"] and not error_log:
        error_log = "Unknown Compile Error (No log returned)"
        
    return result["success"], error_log

def load_dataset_from_json(json_path: str) -> DatasetDict:
    """
    读取 JSON 列表并封装为 HF DatasetDict ('train' split)。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)
    return DatasetDict({'train': dataset})



def split_at_last_prove(s: str) -> tuple[str, str]:
    """
    以最后一个 "prove:" 为界分割字符串。
    """
    last_prove_index = s.rfind('prove:')
    if last_prove_index == -1:
        return (s, '')
    return (s[:last_prove_index], s[last_prove_index:])

def extract_latex_claim(tex: str) -> str | None:
    """
    提取 LaTeX 中 \\begin{claim} 和 \\begin{proof} 之间的内容。
    """
    pattern = re.compile(r"\\begin\{claim\}(.*?)\\begin\{proof\}", re.DOTALL)
    m = pattern.search(tex)
    return m.group(1).strip() if m else None

def convert_litex_to_latex(litex_code: str) -> dict:
    """
    调用核心库将 Litex 转换为 LaTeX。
    """
    try:
        result = pylitex.convert_to_latex(litex_code.replace("\r\n", "\n"))
        claim_content = extract_latex_claim(result["message"])
        if claim_content is not None:
            return {"success": True, "message": claim_content}
        else:
            return {"success": False, "message": "No claim environment found in the LaTeX output."}
    except subprocess.CalledProcessError as e:
        return {"success": False, "message": e.stderr}
    except FileNotFoundError:
        return {"success": False, "message": "Litex command not found."}

def construct_verification_prompt(row: dict[str, str]) -> list[dict[str, str]] | None:
    """
    生成用于验证 LaTeX 代码是否解决问题的 Prompt。
    已根据要求将限制条件合并为一大段并打乱语序。
    """
    topic = row["nl_problem"]
    litex_code = row["formal_statement"]
    
    try:
        latex_result = convert_litex_to_latex(litex_code)
        if not latex_result["success"]:
            return None
        latex_code = latex_result["message"]
    except Exception:
        return None

    # 合并后的 Prompt 内容
    system_prompt = "You are a knowledgeable assistant skilled in evaluating LaTeX code for mathematical and logical correctness. You should follow the user's instructions carefully and provide accurate assessments based on the provided LaTeX code and topic. you should answer \"Yes\" or \"No\" only."
    
    # 将多个 restrict 合并为一段，并打乱了原有的顺序
    restrictions_block = (
        "Consider these restrictions: You must answer \"No\" if the same answer shown both before and after the $\\Rightarrow$ symbol. "
        "However, you should answer \"Yes\" if the LaTeX code is transforming the polynomial to another form for those polynomial transformation or simplification problems, "
        "or if it is solving for a variable or simplifying an expression for those easy math algebra problems. "
        "Additionally, answer \"Yes\" if the LaTeX code is translating the conceptions only for those basic math conceptions, "
        "or if it is using different symbol to describe the vars in the topic (like \"x\", \"y\", \"z\") while representing the same calculation relationship. "
        "Even if the code is directly providing the final answer or formal_statement to the problem (for obvious math problems), "
        "or if it is clearly and unambiguously attempting to describe or solve the given topic, you should answer \"Yes\"."
    )

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": restrictions_block},
        {"role": "user", "content": f"Here is the topic and the LaTeX code:\nTopic:\n{topic}\n\nLaTeX code:\n```\n{latex_code}\n```"},
        {"role": "user", "content": "Is the LaTeX code describe the topic? Answer \"Yes\" or \"No\" only."}
    ]

    return prompt

def get_model_list() -> list[str]:
    return ["qwen-max", "qwen-plus"]

def query_llm(info: tuple[str, list[dict[str, str]]]) -> str | None:
    (model, prompt) = info
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=prompt,
            timeout=30,
        )
        return completion.choices[0].message.content

    except APIStatusError as e:
        if e.status_code == 400 and "balance" in str(e).lower():
            print("Token 配额已耗尽！")
        elif e.status_code == 429:
            print("请求过于频繁 (Rate Limit)。")
        else:
            print(f"API 状态错误: {e.status_code}")
        return None
    except Exception as e:
        print(f"请求错误: {e}")
        return None

def verify_litex_semantics(row: dict[str, str]):
    """
    使用 LLM 投票验证语义正确性。
    """
    prompt = construct_verification_prompt(row)
    if prompt is None:
        return {**row, "actual": "No"}

    results = []
    # 两轮投票
    for _ in range(2):
        for model in get_model_list():
            result = query_llm((model, prompt))
            results.append(result)

    answer = "Yes" if "Yes" in results else "No"
    return {**row, "actual": answer}

def evaluate_litex_instance(row: dict[str, str]):
    """
    综合验证 Litex 代码的语法与语义。
    """
    id_ = row.get("id", "")
    nl_problem = row.get("nl_problem", "")
    formal_statement = row.get("formal_statement", "")

    # 1. 语法验证
    syntax_success, error_log = verify_litex_syntax(formal_statement)
    
    # 2. 语义验证 (仅当语法正确时)
    semantic_correctness = False
    if syntax_success:
        semantic_result = verify_litex_semantics(row)
        semantic_correctness = (semantic_result.get("actual", "").strip().lower() == "yes")

    return {
        "id": id_,
        "nl_problem": nl_problem,
        "formal_statement": formal_statement,
        "semantic_correctness": semantic_correctness,
        "grammar_correctness": syntax_success,
        "error_log": error_log,
        "correctness": syntax_success and semantic_correctness,
    }

def evaluate_dataset_file(file_path: str) -> dict:
    """
    批量评估 JSONL 文件中的样本。
    """
    results = []
    total = 0
    correct = 0

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    for line in tqdm.tqdm(lines, desc="Evaluating Litex correctness", unit="sample"):
        data = json.loads(line)
        row = {
            "id": data.get("id", ""),
            "nl_problem": data.get("nl_problem", ""),
            "formal_statement": data.get("formal_code", ""),
        }
        result = evaluate_litex_instance(row)
        score = 1 if result["correctness"] else 0
        correct += score
        total += 1
        results.append({**result, "score": score})

    accuracy = round((correct / total) * 100, 2) if total > 0 else 0.0
    print(f"\n✅ Evaluated {total} samples. Score: {correct}/{total} = {accuracy}%")

    return {"score": accuracy, "results": results}

def format_training_example(example):
    """
    构造 SFT 训练数据。
    (保持 Prompt 原始内容不变)
    """
    nl_problem = example["nl_problem"]
    formal_statement = example["formal_statement"]
    id_ = example["id"]
    claim, prove = split_at_last_prove(formal_statement)

    LITEX_COMPLEX_EXAMPLE = """have self_complex set

fn sc(x, y R) self_complex
fn sc_mul(x, y self_complex) self_complex

know:
    forall x, y, a, b R:
        sc_mul(sc(x, y), sc(a, b)) = sc(x * a - y * b, x * b + y * a)

claim:
    forall q1, q2, e1, e2 R:
        q1 = 11
        q2 = -5
        e1 = 11
        e2 = 5
        =>:
            sc_mul(sc(q1, q2), sc(e1, e2)) = sc(146, 0)
    prove:
        sc_mul(sc(q1, q2), sc(e1, e2)) = sc_mul(sc(11, -5), sc(11, 5))
        sc_mul(sc(11, -5), sc(11, 5)) = sc(11 * 11 - (-5) * 5, 11 * 5 + (-5) * 11)
        sc_mul(sc(11, -5), sc(11, 5)) = sc(121 + 25, 55 - 55)
        sc_mul(sc(11, -5), sc(11, 5)) = sc(146, 0)"""

    user_input = f"""You are a formal verification expert specialized in the Litex language. Your task is to translate a natural language mathematical problem into a complete and syntactically correct Litex formal statement.

### Language Rules you should strictly follow:
1.  **Custom Definitions**: If the problem involves concepts not built into standard arithmetic (e.g., complex numbers, vectors, custom operators), you MUST define them using `have` (for types), `fn` (for functions), and `know` (for axioms/definitions) **before** the `claim:` block.
2.  **Structure**: The output must contain a `claim:` section (the proposition) and a `prove:` section (the derivation).
3.  **Indentation**: Use strict 4-space indentation for logic nesting.
4.  **Typing**: Explicitly quantify all variables (e.g., `forall x R`, `exists n Z`) in the `claim`.

### Reference Example (Learning from Context)
Here is an example showing how to define custom structures and prove a claim: 
{LITEX_COMPLEX_EXAMPLE}
### Problem
    {nl_problem}"""
    
    data = {
        "messages": [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": formal_statement}
        ],
        "user_input": user_input,
        "question": claim,
        "formal_statement": formal_statement,
        "id": id_,
        "nl_problem": nl_problem,
    }
    return data