# model_Adjust
基于 Ollama 模型的情感分析

本项目演示了如何使用 Ollama 的 Gemma 模型对 GLUE 数据集中的 SST-2 数据集进行情感分析。代码实现了数据集加载、预测请求以及结果评估的过程。

环境要求

Python 3.x

安装以下依赖：

pip install transformers datasets requests

设置

下载数据集
本项目使用 GLUE 基准中的 SST-2 数据集，代码如下：

from datasets import load_dataset
dataset = load_dataset("glue", "sst2", split="test[:20]")


调用 Ollama 模型进行预测
使用 Ollama 模型 API 进行情感预测。代码如下：

import requests
import time

def query_ollama(prompt, model="gemma:2b-instruct"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # 降低随机性
            "num_predict": 50    # 限制输出长度
        }
    }
    response = requests.post(url, json=data)
    return response.text


评估预测结果
预测结果包括情感的正确与否，并计算准确率，代码如下：

results = []  # 存储预测结果
successful_requests = 0

# 统计有效预测与错误预测
valid_predictions = [r for r in results if r['predicted_sentiment'] is not None]
correct_predictions = sum(1 for r in valid_predictions if r['correct'])
total_valid = len(valid_predictions)

print(f"有效预测准确率: {correct_predictions/total_valid:.2%}")
代码里面给出了ollama本地调用的几个模型的准确率，由于模型较少以及训练的条数有限，准确率最高45%

Qwen3 Model Instruction Tuning

本项目展示了如何使用 Qwen3 模型进行指令微调，以进行情感分析任务。通过结合 LoRA（Low-Rank Adaptation）和 4-bit 量化 技术，本项目实现了高效的微调流程，并通过使用 Unsloth 框架优化了内存和计算资源的使用。

项目结构

imdb_qwen3_instruct2.py：主脚本，加载 IMDB 数据集，进行模型微调并执行情感分析任务。

instruction-qwen2-5-1-5b-instruct.ipynb：Jupyter notebook，包含更多的模型微调步骤和推理过程。

环境配置
安装依赖

安装 Unsloth 框架：

pip install unsloth
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git


安装其他依赖：

pip install torch transformers datasets sklearn pandas numpy evaluate

配置 HuggingFace 认证

如果需要从 HuggingFace Hub 下载模型，可以使用以下代码进行认证：

from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
login(hf_token)

数据集

项目使用 IMDB 数据集进行情感分析。数据集分为训练集、验证集和测试集，数据被格式化为 Alpaca 风格的指令任务。

数据加载与处理
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

train_df = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test_df = pd.read_csv("./corpus/imdb/testData.tsv", header=0, delimiter="\t", quoting=3)

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=3407)

train_dict = {'label': train_df["sentiment"], 'text': train_df['review']}
val_dict = {'label': val_df["sentiment"], 'text': val_df['review']}
test_dict = {"text": test_df['review']}

train_dataset = Dataset.from_dict(train_dict)
val_dataset = Dataset.from_dict(val_dict)
test_dataset = Dataset.from_dict(test_dict)
![alt text](https://github.com/viaviachris/model_Adjust/blob/main/%E7%BC%BA%E5%A4%B1%E7%8E%87.png)
模型加载与微调
加载 Qwen3 模型
from unsloth import FastLanguageModel

model_name = "unsloth/Qwen3-4B-base"
model, tokenizer = FastLanguageModel.from_pretrained(model_name, load_in_4bit=True, max_seq_length=2048)

PEFT (LoRA) 设置
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_gradient_checkpointing="unsloth",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

微调过程
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs_qwen",
    per_device_train_batch_size=16,
    warmup_steps=100,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
)

trainer.train()



推理与结果保存

微调后的模型可以用于对测试集进行推理：

test_texts = test_dataset['text']
predictions = []

for review_text in test_texts:
    prompt = inference_prompt.format(review_text)
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=3, eos_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        response_part = generated_text.split("### Response:")[1].strip().lower()
        if "positive" in response_part:
            predictions.append(1)
        else:
            predictions.append(0)
    except (IndexError, AttributeError):
        predictions.append(0)

result_output = pd.DataFrame(data={"id": test_ids, "sentiment": predictions})
result_output.to_csv("../results/qwen3_4b_instruct_unsloth.csv", index=False)
后面增加训练条数，使用phi-3.5-mini-instruct,损失率有效降低了但是准确率并没有明显变化，后续换几个模型进行调试，调增下lora参数
![alt text](https://github.com/viaviachris/model_Adjust/blob/main/phi-3.5%E7%BC%BA%E5%A4%B1%E7%8E%87.png)

总结

本项目展示了如何通过 LoRA 和量化技术对 Qwen3 模型进行高效的微调，以执行情感分析任务。使用 Unsloth 框架优化了内存和计算效率，使得在较低的硬件要求下也能进行有效的训练和推理。由于显卡和内存限制先用小模型进行微调，只训练了部分数据，效果不是太好后面训练了phi-3.5-mini-instruct,增加训练条数提升准确率
