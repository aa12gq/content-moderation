#!/bin/bash

# 内容审核平台本地开发环境启动脚本

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    echo "可以使用以下命令安装:"
    echo "brew install python"
    exit 1
fi

if ! command -v pip3 &> /dev/null; then
    echo "错误: 未找到pip3，请先安装pip3"
    echo "可以使用以下命令安装:"
    echo "brew install python"
    exit 1
fi

# 检查Go环境
if ! command -v go &> /dev/null; then
    echo "错误: 未找到Go，请先安装Go"
    echo "可以使用以下命令安装:"
    echo "brew install go"
    exit 1
fi

# 检查protoc
if ! command -v protoc &> /dev/null; then
    echo "错误: 未找到protoc，请先安装protoc"
    echo "可以使用以下命令安装:"
    echo "brew install protobuf"
    exit 1
fi

# 确保目录结构
mkdir -p models
mkdir -p logs
mkdir -p scripts

# 创建并激活Python虚拟环境
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv venv
fi

echo "激活Python虚拟环境..."
source venv/bin/activate

# 准备Python环境
echo "检查Python依赖..."
pip install -r requirements.txt

# 生成proto文件
echo "生成proto文件..."
chmod +x scripts/generate_proto.sh
./scripts/generate_proto.sh

# 启动NLP服务（后台运行）
echo "启动NLP服务..."
python server/python/nlp_service.py > logs/nlp_service.log 2>&1 &
NLP_PID=$!
echo "NLP服务已启动 (PID: $NLP_PID)"

# 等待NLP服务启动
echo "等待NLP服务就绪..."
sleep 5

# 编译并启动Go RPC服务
echo "启动RPC服务..."
cd server/go
go mod tidy
go run main.go > ../../logs/rpc_service.log 2>&1 &
RPC_PID=$!
cd ../..
echo "RPC服务已启动 (PID: $RPC_PID)"

# 等待RPC服务启动
echo "等待RPC服务就绪..."
sleep 3

echo "服务已启动!"
echo "- NLP服务: localhost:50052 (PID: $NLP_PID)"
echo "- RPC服务: localhost:50051 (PID: $RPC_PID)"
echo ""
echo "使用以下命令运行测试客户端:"
echo "source venv/bin/activate && python client/client.py"
echo ""
echo "使用 Ctrl+C 停止服务"

# 捕获SIGINT信号（Ctrl+C）
trap "echo '正在停止服务...'; kill $NLP_PID $RPC_PID; deactivate; echo '服务已停止'; exit" INT

# 保持脚本运行
wait $NLP_PID $RPC_PID 