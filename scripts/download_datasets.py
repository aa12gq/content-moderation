import os
import json
import gzip
import shutil
import requests
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path

# 配置
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
OUTPUT_FILE = os.path.join(DATA_DIR, 'insult_train.json')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')

# 数据集配置
DATASETS = {
    'weibo_sentiment': {
        'url': 'https://huggingface.co/datasets/weibo_sentiment_100k/resolve/main/weibo_sentiment_100k.csv',
        'backup_url': 'https://raw.githubusercontent.com/CLUEbenchmark/CLUE/master/datasets/weibo_sentiment_100k/weibo_sentiment_100k.csv',
        'type': 'csv',
        'columns': ['label', 'text']
    },
    'chid': {
        'url': 'https://huggingface.co/datasets/chinese-insult-detection/resolve/main/train.json',
        'backup_url': 'https://raw.githubusercontent.com/ShannonAI/Chinese-Insult-Detection/master/data/train.json',
        'type': 'json'
    }
}

# 示例数据（用于测试和补充）
EXAMPLE_DATA = [
    # 直接侮辱
    {
        "text": "你真是个白痴",
        "label": 1
    },
    {
        "text": "你太蠢了",
        "label": 1
    },
    {
        "text": "你是个笨蛋",
        "label": 1
    },
    {
        "text": "你这个废物",
        "label": 1
    },
    {
        "text": "你脑子有问题吧",
        "label": 1
    },
    
    # 间接侮辱
    {
        "text": "就你这水平还敢出来丢人",
        "label": 1
    },
    {
        "text": "你连这个都不会，真是可笑",
        "label": 1
    },
    {
        "text": "这么简单的问题都不会，你是猪吗",
        "label": 1
    },
    
    # 讽刺性侮辱
    {
        "text": "你可真是个大聪明",
        "label": 1
    },
    {
        "text": "你真是太厉害了，连这个都不会",
        "label": 1
    },
    
    # 正常对话
    {
        "text": "今天天气真好",
        "label": 0
    },
    {
        "text": "我喜欢这个电影",
        "label": 0
    },
    {
        "text": "这个餐厅的菜很好吃",
        "label": 0
    },
    {
        "text": "请问这个怎么用",
        "label": 0
    },
    {
        "text": "谢谢你的帮助",
        "label": 0
    },
    
    # 中性评价
    {
        "text": "这个产品一般般",
        "label": 0
    },
    {
        "text": "我觉得还可以改进",
        "label": 0
    },
    {
        "text": "这个方案需要再考虑一下",
        "label": 0
    },
    
    # 礼貌的批评
    {
        "text": "这个设计可能不太合适",
        "label": 0
    },
    {
        "text": "建议可以换个方式",
        "label": 0
    },
    {
        "text": "这个想法需要再完善一下",
        "label": 0
    }
]

def download_file(url: str, save_path: str, backup_url: str = None) -> bool:
    """下载文件，如果主链接失败则尝试备用链接"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(save_path, 'wb') as f, tqdm(
            desc=f"下载 {os.path.basename(save_path)}",
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)
        return True
    except Exception as e:
        print(f"主链接下载失败: {e}")
        if backup_url:
            print(f"尝试使用备用链接: {backup_url}")
            try:
                response = requests.get(backup_url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024
                
                with open(save_path, 'wb') as f, tqdm(
                    desc=f"下载 {os.path.basename(save_path)} (备用链接)",
                    total=total_size,
                    unit='iB',
                    unit_scale=True
                ) as pbar:
                    for data in response.iter_content(block_size):
                        pbar.update(len(data))
                        f.write(data)
                return True
            except Exception as e:
                print(f"备用链接下载也失败: {e}")
                return False
        return False

def process_weibo_sentiment(data_path: str) -> List[Dict[str, Any]]:
    """处理微博情感数据集"""
    processed_data = []
    try:
        df = pd.read_csv(data_path)
        for _, row in df.iterrows():
            text = row['text']
            label = row['label']
            # 将情感标签转换为侮辱性标签（这里需要根据实际数据调整）
            insult_label = 1 if label == 0 else 0  # 假设负面情感为侮辱性
            processed_data.append({
                "text": text,
                "label": insult_label
            })
    except Exception as e:
        print(f"处理微博情感数据失败: {e}")
    return processed_data

def process_chid(data_path: str) -> List[Dict[str, Any]]:
    """处理ChID数据集"""
    processed_data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                text = item['text']
                label = item['label']
                processed_data.append({
                    "text": text,
                    "label": label
                })
    except Exception as e:
        print(f"处理ChID数据失败: {e}")
    return processed_data

def merge_datasets(datasets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """合并多个数据集"""
    merged_data = []
    for dataset in datasets:
        merged_data.extend(dataset)
    return merged_data

def save_dataset(data: List[Dict[str, Any]], output_file: str):
    """保存处理后的数据集"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # 创建临时目录
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 下载和处理数据集
    processed_datasets = []
    
    # 添加示例数据
    processed_datasets.append(EXAMPLE_DATA)
    print(f"\n添加示例数据: {len(EXAMPLE_DATA)} 条")
    
    for name, config in DATASETS.items():
        print(f"\n处理数据集: {name}")
        temp_file = os.path.join(TEMP_DIR, f"{name}.temp")
        
        # 下载数据集
        backup_url = config.get('backup_url')
        if download_file(config['url'], temp_file, backup_url):
            # 根据数据集类型处理
            if config['type'] == 'csv':
                data = process_weibo_sentiment(temp_file)
            elif config['type'] == 'json':
                data = process_chid(temp_file)
            else:
                print(f"未知的数据集类型: {config['type']}")
                continue
                
            processed_datasets.append(data)
            print(f"成功处理 {len(data)} 条数据")
        else:
            print(f"数据集 {name} 下载失败，跳过处理")
    
    if not processed_datasets:
        print("没有成功下载任何数据集，请检查网络连接或更新下载链接")
        return
    
    # 合并数据集
    print("\n合并数据集...")
    merged_data = merge_datasets(processed_datasets)
    
    # 保存处理后的数据集
    print(f"\n保存数据集到: {OUTPUT_FILE}")
    save_dataset(merged_data, OUTPUT_FILE)
    
    # 清理临时文件
    shutil.rmtree(TEMP_DIR)
    
    print(f"\n处理完成！共处理 {len(merged_data)} 条数据")
    print(f"数据集已保存到: {OUTPUT_FILE}")

if __name__ == '__main__':
    main() 