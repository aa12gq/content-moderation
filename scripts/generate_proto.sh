#!/bin/bash

# 生成 proto 文件

# 设置目录
PROTO_DIR="proto"
GO_OUT_DIR="server/go"
PYTHON_OUT_DIR="server/python/proto/gen"

# 创建输出目录
mkdir -p $GO_OUT_DIR
mkdir -p $PYTHON_OUT_DIR

# 生成 Go 代码
protoc --go_out=$GO_OUT_DIR \
       --go_opt=module=content-moderation \
       --go-grpc_out=$GO_OUT_DIR \
       --go-grpc_opt=module=content-moderation \
       $PROTO_DIR/contentmoderation.proto

# 确保在虚拟环境中运行 Python
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 生成 Python 代码
python3 -m grpc_tools.protoc \
       --proto_path=$PROTO_DIR \
       --python_out=$PYTHON_OUT_DIR \
       --grpc_python_out=$PYTHON_OUT_DIR \
       $PROTO_DIR/contentmoderation.proto 