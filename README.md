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
