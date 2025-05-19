FROM golang:1.16-alpine AS builder

WORKDIR /app

# 复制依赖配置
COPY server/go/go.mod .
COPY server/go/go.sum .

# 下载依赖
RUN go mod download

# 复制源代码
COPY proto/ /app/proto/
COPY server/go/ /app/server/go/

# 使用protoc生成gRPC代码
RUN apk add --no-cache protobuf-dev
RUN go get -u google.golang.org/protobuf/cmd/protoc-gen-go
RUN go get -u google.golang.org/grpc/cmd/protoc-gen-go-grpc
RUN mkdir -p /app/pb
RUN protoc --go_out=/app/pb --go-grpc_out=/app/pb --go_opt=paths=source_relative --go-grpc_opt=paths=source_relative /app/proto/contentmoderation.proto

# 编译应用
WORKDIR /app/server/go
RUN go build -o rpc_server

# 多阶段构建，创建更小的运行镜像
FROM alpine:latest

WORKDIR /app

# 复制编译好的二进制文件
COPY --from=builder /app/server/go/rpc_server /app/

# 设置环境变量
ENV SERVICE_PORT=50051
ENV NLP_SERVICE_ADDR=nlp-service:50052

# 暴露端口
EXPOSE 50051

# 启动服务
CMD ["./rpc_server"] 