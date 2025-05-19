import torch
import json
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    def __init__(self, model_dir: str = "./models", rules_file: str = "./rules/rules.json"):
        """
        初始化NLP审核引擎
        
        Args:
            model_dir: 模型目录
            rules_file: 规则文件路径
        """
        self.model_dir = model_dir
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            logger.info("成功加载BERT分词器")
            
            # 加载模型
            self.models = {}
            self._load_models()
            
            # 加载规则引擎
            self.rule_engine = RuleEngine(rules_file)
            
        except Exception as e:
            logger.error(f"初始化NLP引擎失败: {e}")
            raise
    
    def _load_models(self):
        """加载所有分类模型"""
        categories = ["spam", "porn", "violence", "political", "insult"]
        
        # 第一次运行时可能没有训练好的模型，使用基础BERT模型替代
        # 实际环境中应该使用针对每个类别微调的模型
        for category in categories:
            try:
                model_path = f"{self.model_dir}/{category}_model"
                # 尝试加载特定模型，如果不存在则使用基础模型
                try:
                    self.models[category] = AutoModelForSequenceClassification.from_pretrained(model_path)
                    logger.info(f"成功加载{category}模型: {model_path}")
                except:
                    # 首次运行时使用基础模型并将其设置为二分类模型
                    logger.warning(f"未找到{category}模型，使用基础BERT模型替代")
                    self.models[category] = AutoModelForSequenceClassification.from_pretrained(
                        "bert-base-chinese", 
                        num_labels=2
                    )
            except Exception as e:
                logger.error(f"加载{category}模型失败: {e}")
                # 出错时继续加载其他模型
                continue
    
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
            categories = list(self.models.keys())
        
        results = []
        
        # 对每个请求的类别进行检测
        for category in categories:
            if category in self.models:
                # 使用对应模型进行预测
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = self.models[category](**inputs)
                        
                    # 获取分类结果
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    confidence = probabilities[0][1].item()  # 假设1是正类（即包含该类内容）
                    detected = confidence > 0.5
                except Exception as e:
                    logger.error(f"模型预测失败: {e}")
                    # 模型预测失败时，设置默认值
                    confidence = 0.0
                    detected = False
                
                # 规则引擎补充
                rule_result = self.rule_engine.check(text, category)
                
                # 融合结果（规则引擎检测到则覆盖模型结果）
                if rule_result["detected"]:
                    detected = True
                    matched_keywords = rule_result["matched_keywords"]
                else:
                    matched_keywords = []
                
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
        model_dir="../../../models", 
        rules_file="../../../rules/rules.json"
    )
    
    # 测试用例
    test_cases = [
        "这是一条正常的消息，不包含任何敏感内容。",
        "点击领取限时优惠，免费获取会员资格！",
        "这个内容包含色情和暴力内容，非常不适合未成年人观看。",
        "这个产品实在太好用了，强烈推荐给大家！"
    ]
    
    # 执行测试
    for i, text in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {text}")
        result = engine.predict(text)
        if result["is_approved"]:
            print("审核结果: 通过")
        else:
            print(f"审核结果: 拒绝 - {result['rejection_reason']}")
            for cat_result in result["category_results"]:
                if cat_result["detected"]:
                    print(f"- {cat_result['category']}: 置信度 {cat_result['confidence']:.2f}")
                    if cat_result["matched_keywords"]:
                        print(f"  匹配关键词: {', '.join(cat_result['matched_keywords'])}") 