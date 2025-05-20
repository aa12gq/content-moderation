# 基于 NLP 的内容审核平台

一个高性能、可扩展的内容审核平台，使用先进的 NLP 技术和规则引擎对文本内容进行智能审核。

## 系统架构

### 整体架构

```
+----------------+     +------------------+     +----------------+
| 客户端应用      | --> | RPC接口层        | --> | NLP审核引擎    |
+----------------+     +------------------+     +----------------+
                                |                      |
                                v                      v
                       +------------------+    +----------------+
                       | 审核规则管理     |    | 模型训练与更新 |
                       +------------------+    +----------------+
                                |                      |
                                v                      v
                       +------------------+    +----------------+
                       | 审核日志与统计   |    | 审核结果存储   |
                       +------------------+    +----------------+
```

### 核心组件

- **RPC 接口层**：使用 gRPC 提供高性能的远程调用接口
- **NLP 审核引擎**：基于 BERT/Transformers 的文本分析与内容识别
- **审核规则管理**：维护和更新审核规则
- **模型训练与更新**：支持模型的持续优化
- **审核日志与统计**：记录审核过程和结果
- **审核结果存储**：存储历史审核数据

## 技术栈

### 后端技术

- **编程语言**：
  - Python (NLP 处理)
  - Go (高性能 RPC 服务)
- **RPC 框架**：gRPC
- **NLP 框架**：PyTorch, HuggingFace Transformers
- **数据库**：
  - MongoDB (非结构化审核数据)
  - Redis (缓存与高频规则)
  - PostgreSQL (结构化数据与统计)

### 部署技术

- Docker + Kubernetes
- CI/CD 流水线

## 快速开始

### 环境要求

- Python 3.8+
- Go 1.16+
- Docker & Docker Compose (可选)
- Kubernetes (可选，用于生产环境)

### 本地安装与运行

1. 克隆仓库

```bash
git clone https://github.com/aa12gq/content-moderation.git
cd content-moderation
```

2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

3. 下载预训练模型（首次运行会自动下载）

4. 启动 NLP 服务

```bash
python server/python/nlp_service.py
```

5. 在另一个终端启动 Go RPC 服务

```bash
cd server/go
go mod tidy
go run main.go
```

6. 运行示例客户端

```bash
python client/client.py
```

### 使用 Docker 运行

```bash
# 构建镜像
docker build -t content-moderation/nlp-service -f deployment/Dockerfile.nlp .
docker build -t content-moderation/rpc-service -f deployment/Dockerfile.go .

# 运行服务
docker run -d --name nlp-service -p 50052:50052 content-moderation/nlp-service
docker run -d --name rpc-service -p 50051:50051 -e NLP_SERVICE_ADDR=host.docker.internal:50052 content-moderation/rpc-service
```

## 支持的审核类别

- 垃圾信息/广告
- 色情内容
- 暴力内容
- 政治敏感
- 侮辱/歧视
- 不良信息
- 自定义类别

## API 使用示例

### Python 客户端

```python
from client.client import ModerationClient

# 创建客户端
client = ModerationClient("localhost:50051")

# 同步审核
result = client.moderate_content("待审核的文本内容")
print(f"审核结果: {'通过' if result['is_approved'] else '拒绝'}")

# 批量审核
batch_contents = [
    {"content": "第一条内容"},
    {"content": "第二条内容"}
]
results = client.moderate_batch_content(batch_contents)

# 关闭连接
client.close()
```

### 直接调用 gRPC 接口

可以使用任何支持 gRPC 的语言直接调用 API，参考`proto/contentmoderation.proto`文件查看接口定义。

### 使用 gRPC 客户端工具

对于开发和测试，你可以使用各种 gRPC 客户端工具：

1. **使用 Postman**:

   - 创建新的 gRPC 请求
   - 输入服务器 URL: `localhost:50051`
   - 服务定义将通过反射自动加载
   - 选择`ContentModerationService/ModerateContent`方法
   - 填写请求参数，例如:
     ```json
     {
       "content_id": "test1",
       "content": "这是测试内容",
       "content_type": "text",
       "categories": ["spam", "porn", "violence"]
     }
     ```

2. **使用 BloomRPC 或 grpcui**:

   - 这些工具也支持服务反射，设置类似

3. **故障排除**:
   - 如果遇到"Could not load server reflection"错误，确保服务已正确启动
   - 检查端口是否正确（默认为 50051）
   - 确认防火墙设置允许访问该端口

## 测试结果示例

以下是系统运行时的实际输出示例，展示了内容审核的效果：

### 正常内容审核

```
内容: 这是一条正常的测试消息，不包含任何敏感内容。
审核结果: 通过
```

### 垃圾广告内容审核

```
内容: 点击领取限时优惠，免费获取会员资格！这是一条广告信息。
审核结果: 拒绝
拒绝原因: 内容违反规则: spam
- spam: 置信度 1.00
  匹配关键词: 免费获取, 点击领取, 限时优惠
```

### 包含敏感词的内容审核

```
内容: 这个内容需要异步处理，包含一些可能的敏感词如'暴力'和'色情'。
审核结果: 拒绝
拒绝原因: 内容违反规则: porn, violence
- porn: 置信度 1.00
  匹配关键词: 色情
- violence: 置信度 1.00
  匹配关键词: 暴力
```

### 批量内容审核

```
第1条内容: 正常内容示例，这条消息应该通过审核。
审核结果: 通过

第2条内容: 点击领取限时优惠，免费获取会员资格！
审核结果: 拒绝
拒绝原因: 内容违反规则: spam
- spam: 置信度 1.00
  匹配关键词: 免费获取, 点击领取, 限时优惠

第3条内容: 这个内容包含暴力描述，如杀人和血腥场景。
审核结果: 拒绝
拒绝原因: 内容违反规则: violence
- violence: 置信度 1.00
  匹配关键词: 暴力, 杀人, 血腥

第4条内容: 这是一条正常的产品评价，质量不错，推荐购买。
审核结果: 通过
```

## 自定义和扩展

### 添加新的审核类别

1. 在`rules/rules.json`中添加新类别的关键词
2. 训练对应类别的分类模型，并放入`models/`目录
3. 重启服务

### 性能优化

- **模型量化**：减小模型体积，提高推理速度
- **批处理**：对请求进行批量处理
- **缓存机制**：对高频内容进行缓存
- **异步处理**：长文本采用异步处理模式

## 模型训练与使用

### 预训练模型

系统默认使用 `hfl/chinese-macbert-base` 作为基础预训练模型，这是一个在中文任务上表现优秀的模型。您也可以选择其他预训练模型：

- `hfl/chinese-roberta-wwm-ext`：更大的模型，效果更好但需要更多资源
- `hfl/chinese-macbert-large`：MacBERT 的大版本
- `IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese`：更新的架构

### 训练数据准备

1. 数据格式：

```json
{
  "category_name": {
    "texts": ["正常内容示例1", "敏感内容示例1", "变体表达示例1"],
    "labels": [0, 1, 1] // 0表示正常，1表示敏感
  }
}
```

2. 数据要求：
   - 每个类别至少需要 1000 条训练数据
   - 正负样本比例建议为 1:3
   - 包含各种变体表达（如谐音、同音字等）
   - 考虑上下文相关的表达

### 模型训练

1. 准备训练数据：

```python
train_data = {
    "insult": {
        "texts": [
            "你真是个笨蛋",
            "像你妈妈怎么样",
            "你太蠢了",
            # ... 更多训练数据
        ],
        "labels": [1, 1, 1]  # 1表示敏感内容
    }
}
```

2. 训练模型：

```python
from server.python.nlp_engine import NLPModerationEngine

# 初始化引擎
engine = NLPModerationEngine(
    model_dir="./models",
    rules_file="./rules/rules.json"
)

# 训练特定类别的模型
engine.train_model(
    category="insult",
    train_texts=train_data["insult"]["texts"],
    train_labels=train_data["insult"]["labels"],
    eval_texts=eval_texts,  # 评估数据
    eval_labels=eval_labels
)
```

3. 训练参数说明：
   - `num_train_epochs`: 训练轮数（默认 3 轮）
   - `per_device_train_batch_size`: 训练批次大小（默认 16）
   - `warmup_steps`: 预热步数（默认 500）
   - `weight_decay`: 权重衰减（默认 0.01）

### 模型评估

1. 使用测试集评估：

```python
# 准备测试数据
test_texts = ["测试文本1", "测试文本2"]
test_labels = [0, 1]

# 评估模型
results = engine.predict(test_texts)
```

2. 评估指标：
   - 准确率（Accuracy）
   - 精确率（Precision）
   - 召回率（Recall）
   - F1 分数

### 模型更新

1. 增量训练：

   - 收集新的训练数据
   - 使用 `train_model` 方法进行增量训练
   - 模型会自动保存到指定目录

2. 模型切换：
   - 将新模型放入 `models/` 目录
   - 重启服务即可使用新模型

### 最佳实践

1. 数据收集：

   - 定期收集用户反馈
   - 关注新出现的变体表达
   - 保持训练数据的多样性

2. 模型优化：

   - 定期评估模型性能
   - 根据评估结果调整训练参数
   - 考虑使用模型集成

3. 部署建议：
   - 使用模型量化减小体积
   - 实现模型版本管理
   - 保持模型更新日志

### 数据采集与标注建议

1. **数据采集**

   - 可通过爬虫采集公开评论、弹幕、论坛等文本数据。
   - 初步用关键词、正则或已有模型筛选疑似敏感内容。
   - 采集时注意合规，避免侵犯隐私。

2. **半自动标注**

   - 利用关键词/模型初筛后，人工快速审核。
   - 推荐使用表格工具（如 Excel）或标注平台（如 doccano）进行批量标注。
   - 标注时建议分为"正常内容""敏感内容""模糊/需讨论"三类，后续可合并。

3. **负样本采集**

   - 负样本（正常内容）要多样化，覆盖日常对话、夸奖、疑问、讨论等。
   - 避免负样本中混入敏感表达。

4. **标注建议**

   - 每个类别建议至少 1000 条样本，正负样本比例 1:3 左右。
   - 多收集网络流行语、谐音、变体、缩写、不同语境下的表达。
   - 标注时可先粗标，后细化。

5. **自动化工具推荐**

   - 可用 Python 脚本辅助采集和初筛。
   - 推荐 doccano 等开源标注平台提升效率。

6. **数据安全与合规**
   - 采集和使用数据时注意遵守相关法律法规。
   - 不要采集或存储敏感个人信息。

## 许可证

MIT License
