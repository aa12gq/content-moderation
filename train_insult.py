import json
from server.python.nlp_engine import NLPModerationEngine

if __name__ == "__main__":
    # 读取训练数据
    with open("data/insult_train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    engine = NLPModerationEngine(model_dir="models", rules_file="rules/rules.json")
    engine.train_model("insult", train_data)
    print("训练完成，模型已保存到 models/insult") 