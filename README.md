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

## 许可证

MIT License
