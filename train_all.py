import os
import json
from server.python.nlp_engine import NLPModerationEngine

def train_category(category, data_path, model_dir="models", rules_file="rules/rules.json"):
    with open(data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    engine = NLPModerationEngine(model_dir=model_dir, rules_file=rules_file)
    engine.train_model(category, train_data)
    print(f"[{category}] 训练完成，模型已保存到 {model_dir}/{category}")

if __name__ == "__main__":
    data_dir = "data"
    print(f"data 目录内容: {os.listdir(data_dir)}", flush=True)
    for filename in os.listdir(data_dir):
        print(f"遍历到文件: {filename}", flush=True)
        if filename.endswith("_train.json"):
            category = filename.replace("_train.json", "")
            data_path = os.path.join(data_dir, filename)
            print(f"准备训练类别: {category}, 数据路径: {os.path.abspath(data_path)}", flush=True)
            train_category(category, data_path)
    print("全部训练流程结束！", flush=True) 