import torch
import json
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentDataset(Dataset):
    """内容审核数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

class RuleEngine:
    """基于关键词的规则引擎"""
    
    def __init__(self, rules_file: str):
        """
        初始化规则引擎
        
        Args:
            rules_file: 规则配置文件路径
        """
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            logger.info(f"成功加载规则文件: {rules_file}")
        except Exception as e:
            logger.error(f"加载规则文件失败: {e}")
            self.rules = {}
    
    def check(self, text: str, category: str) -> Dict[str, Any]:
        """
        检查文本是否包含指定类别的关键词
        
        Args:
            text: 待检查的文本
            category: 检查的类别
            
        Returns:
            包含检测结果的字典
        """
        if category not in self.rules:
            return {"detected": False, "matched_keywords": []}
        
        matched_keywords = []
        for keyword in self.rules[category]:
            if keyword in text:
                matched_keywords.append(keyword)
        
        return {
            "detected": len(matched_keywords) > 0,
            "matched_keywords": matched_keywords
        }


class NLPModerationEngine:
    """NLP内容审核引擎"""
    
    def __init__(self, model_dir: str = "models", rules_file: str = "rules/rules.json"):
        """
        初始化NLP审核引擎
        
        Args:
            model_dir: 模型目录
            rules_file: 规则文件路径
        """
        self.model_dir = model_dir
        self.rules_file = rules_file
        self.models = {}
        self.rules = {}
        
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 加载分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
            logger.info("成功加载RoBERTa-wwm-ext分词器")
        except Exception as e:
            logger.error(f"加载分词器失败: {e}")
            raise
        
        # 加载规则
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            logger.info(f"成功加载规则文件: {rules_file}")
        except Exception as e:
            logger.error(f"加载规则文件失败: {e}")
            raise
            
        # 初始化规则引擎
        self.rule_engine = RuleEngine(rules_file)
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """加载所有分类模型"""
        # 只加载已经训练好的模型，避免使用未训练的基础模型产生误报
        categories = ["insult"]  # 目前只有insult模型经过训练
        
        # 使用 RoBERTa-wwm-ext 作为基础模型
        base_model = "hfl/chinese-roberta-wwm-ext"
        
        for category in categories:
            try:
                model_path = f"{self.model_dir}/{category}_model"
                # 尝试加载特定模型，如果不存在则使用基础模型
                try:
                    self.models[category] = AutoModelForSequenceClassification.from_pretrained(model_path)
                    logger.info(f"成功加载{category}模型: {model_path}")
                except:
                    # 首次运行时使用基础模型并将其设置为二分类模型
                    logger.warning(f"未找到{category}模型，使用RoBERTa-wwm-ext基础模型替代")
                    self.models[category] = AutoModelForSequenceClassification.from_pretrained(
                        base_model, 
                        num_labels=2
                    )
            except Exception as e:
                logger.error(f"加载{category}模型失败: {e}")
                continue
    
    def train_model(self, category: str, train_data: List[Dict[str, Any]], output_dir: str = None):
        """训练特定类别的模型"""
        if not train_data:
            logger.error(f"没有训练数据用于类别 {category}")
            return False
        
        # 如果该类别模型未初始化，则初始化
        if category not in self.models:
            try:
                self.models[category] = AutoModelForSequenceClassification.from_pretrained(
                    "hfl/chinese-roberta-wwm-ext", num_labels=2
                )
                logger.info(f"已初始化{category}类别的基础模型")
            except Exception as e:
                logger.error(f"初始化{category}模型失败: {e}")
                return False
        
        # 准备训练数据
        texts = [item['text'] for item in train_data]
        labels = [item['label'] for item in train_data]
        
        # 创建数据集
        dataset = ContentDataset(texts, labels, self.tokenizer)
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir or f"models/{category}",
            num_train_epochs=5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=1000,
            weight_decay=0.01,
            logging_dir=f"logs/{category}",
            logging_steps=100,
            save_steps=500
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.models[category],
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,  # 这里应该使用验证集
            compute_metrics=self._compute_metrics
        )
        
        # 开始训练
        logger.info(f"开始训练 {category} 模型...")
        trainer.train()
        
        # 保存模型
        model_path = output_dir or f"models/{category}"
        trainer.save_model(model_path)
        logger.info(f"模型已保存到 {model_path}")
        
        return True

    def _compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def predict(self, text: str, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        预测文本内容类别
        
        Args:
            text: 待审核的文本内容
            categories: 指定需要检测的类别，默认为全部类别
            
        Returns:
            审核结果字典
        """
        if categories is None:
            # 只使用已加载的模型和规则中的类别
            model_categories = set(self.models.keys())
            rule_categories = set(self.rules.keys())
            categories = list(model_categories.union(rule_categories))
        
        results = []
        
        # 对每个请求的类别进行检测
        for category in categories:
            detected = False
            confidence = 0.0
            matched_keywords = []
            
            # 规则引擎检查
            rule_result = self.rule_engine.check(text, category)
            if rule_result["detected"]:
                detected = True
                confidence = 1.0
                matched_keywords = rule_result["matched_keywords"]
            
            # 模型预测（仅当规则引擎未检测到且模型已加载时）
            elif category in self.models:
                try:
                    # 使用更长的序列长度以捕获更多上下文
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True,
                        add_special_tokens=True
                    )
                    
                    with torch.no_grad():
                        outputs = self.models[category](**inputs)
                        
                    # 获取分类结果
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    confidence = probabilities[0][1].item()  # 假设1是正类（即包含该类内容）
                    
                    # 使用更严格的阈值
                    detected = confidence > 0.9  # 提高阈值以减少误报
                    
                except Exception as e:
                    logger.error(f"模型预测失败: {e}")
                    confidence = 0.0
                    detected = False
            
            results.append({
                "category": category,
                "detected": detected,
                "confidence": confidence,
                "matched_keywords": matched_keywords
            })
        
        # 整合最终结果
        is_approved = not any(r["detected"] for r in results)
        rejection_reason = ""
        if not is_approved:
            detected_categories = [r["category"] for r in results if r["detected"]]
            rejection_reason = f"内容违反规则: {', '.join(detected_categories)}"
        
        return {
            "is_approved": is_approved,
            "category_results": results,
            "rejection_reason": rejection_reason,
            "confidence_score": max([r["confidence"] for r in results], default=0)
        }


# 测试代码
if __name__ == "__main__":
    # 初始化引擎
    engine = NLPModerationEngine(
        model_dir="models",
        rules_file="rules/rules.json"
    )
    
    # 准备训练数据
    train_data = [
        # 正常内容
        {"text": "今天天气真好", "label": 0},
        {"text": "我喜欢看电影", "label": 0},
        {"text": "这个产品很好用", "label": 0},
        {"text": "谢谢你的帮助", "label": 0},
        {"text": "祝你生日快乐", "label": 0},
        
        # 敏感内容
        {"text": "你太蠢了", "label": 1},
        {"text": "你这个白痴", "label": 1},
        {"text": "你是个智障", "label": 1},
        {"text": "你太笨了", "label": 1},
        {"text": "你真是个废物", "label": 1},
        
        # 变体表达
        {"text": "你太c了", "label": 1},  # 拼音首字母
        {"text": "你太2了", "label": 1},  # 数字谐音
        {"text": "你太菜了", "label": 1},  # 同义词
        {"text": "你太弱了", "label": 1},  # 同义词
        {"text": "你太差劲了", "label": 1},  # 同义词
        
        # 上下文相关
        {"text": "你连这个都不会，太笨了", "label": 1},
        {"text": "这么简单的问题都答错，你太蠢了", "label": 1},
        {"text": "你连这个都做不好，真是个废物", "label": 1},
        {"text": "这么基础的东西都不会，你太差劲了", "label": 1},
        {"text": "你连这个都搞不定，太弱了", "label": 1}
    ]
    
    # 训练模型
    engine.train_model("insult", train_data)
    
    # 测试模型
    test_texts = [
        "今天天气真好",
        "你太蠢了",
        "你太c了",
        "你连这个都不会，太笨了",
        "这个产品很好用",
        "你太菜了",
        "谢谢你的帮助",
        "你太弱了",
        "祝你生日快乐",
        "你太差劲了"
    ]
    
    print("\n测试模型效果:")
    for text in test_texts:
        result = engine.predict("insult", text)
        print(f"\n文本: {text}")
        print(f"预测结果: {result}") 