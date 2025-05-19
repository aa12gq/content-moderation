#!/bin/bash

# 生成protobuf代码的脚本

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 创建输出目录
mkdir -p proto/gen
mkdir -p server/go/pb

# 生成Go代码
protoc --go_out=server/go/pb --go-grpc_out=server/go/pb --go_opt=paths=source_relative --go-grpc_opt=paths=source_relative proto/contentmoderation.proto

# 生成Python代码
python -m grpc_tools.protoc \
  --proto_path=proto \
  --python_out=server/python \
  --grpc_python_out=server/python \
  proto/contentmoderation.proto

python -m grpc_tools.protoc \
  --proto_path=proto \
  --python_out=client \
  --grpc_python_out=client \
  proto/contentmoderation.proto

echo "Protobuf 代码生成完成" 