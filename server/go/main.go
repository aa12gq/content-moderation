package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	pb "content-moderation/proto/gen"
)

var (
	port       = flag.Int("port", 50051, "RPC服务器端口")
	nlpAddress = flag.String("nlp_address", "localhost:50052", "NLP服务地址")
)

// 异步结果缓存
type resultCache struct {
	mu      sync.RWMutex
	results map[string]*pb.ModerationResponse
}

// 创建缓存
func newResultCache() *resultCache {
	return &resultCache{
		results: make(map[string]*pb.ModerationResponse),
	}
}

// 保存结果
func (c *resultCache) set(id string, result *pb.ModerationResponse) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.results[id] = result
	// 设置过期时间，避免内存泄漏（实际项目中应该使用更健壮的方法）
	go func() {
		time.Sleep(30 * time.Minute)
		c.mu.Lock()
		delete(c.results, id)
		c.mu.Unlock()
	}()
}

// 获取结果
func (c *resultCache) get(id string) (*pb.ModerationResponse, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	result, ok := c.results[id]
	return result, ok
}

// 内容审核服务实现
type contentModerationServer struct {
	pb.UnimplementedContentModerationServiceServer
	nlpClient pb.ContentModerationServiceClient
	cache     *resultCache
}

// 同步内容审核
func (s *contentModerationServer) ModerateContent(ctx context.Context, req *pb.ModerationRequest) (*pb.ModerationResponse, error) {
	// 记录请求
	log.Printf("收到同步内容审核请求: %s\n", req.ContentId)

	// 设置RPC超时
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// 转发到NLP服务
	resp, err := s.nlpClient.ModerateContent(ctx, req)
	if err != nil {
		log.Printf("NLP服务调用失败: %v\n", err)
		return nil, status.Errorf(codes.Internal, "内部服务调用失败: %v", err)
	}

	log.Printf("内容 %s 审核结果: %v\n", req.ContentId, resp.IsApproved)
	return resp, nil
}

// 异步内容审核
func (s *contentModerationServer) ModerateContentAsync(ctx context.Context, req *pb.ModerationRequest) (*pb.AsyncModerationResponse, error) {
	log.Printf("收到异步内容审核请求: %s\n", req.ContentId)

	// 设置RPC超时
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// 转发到NLP服务
	resp, err := s.nlpClient.ModerateContentAsync(ctx, req)
	if err != nil {
		log.Printf("NLP服务异步调用失败: %v\n", err)
		return nil, status.Errorf(codes.Internal, "内部服务调用失败: %v", err)
	}

	log.Printf("异步请求已提交: %s\n", resp.RequestId)
	return resp, nil
}

// 获取异步审核结果
func (s *contentModerationServer) GetModerationResult(ctx context.Context, req *pb.ResultRequest) (*pb.ModerationResponse, error) {
	log.Printf("获取异步审核结果: %s\n", req.RequestId)

	// 先检查本地缓存
	if result, found := s.cache.get(req.RequestId); found {
		log.Printf("从缓存获取结果: %s\n", req.RequestId)
		return result, nil
	}

	// 设置RPC超时
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	// 从NLP服务获取结果
	resp, err := s.nlpClient.GetModerationResult(ctx, req)
	if err != nil {
		st, ok := status.FromError(err)
		if !ok {
			log.Printf("获取结果失败: %v\n", err)
			return nil, status.Errorf(codes.Internal, "获取结果失败: %v", err)
		}

		// 根据具体错误类型返回相应状态
		switch st.Code() {
		case codes.ResourceExhausted:
			// 处理中，返回同样的错误
			return nil, status.Errorf(codes.ResourceExhausted, "处理中，请稍后再试")
		case codes.NotFound:
			return nil, status.Errorf(codes.NotFound, "未找到请求ID: %s", req.RequestId)
		default:
			return nil, err
		}
	}

	// 结果成功获取，保存到缓存
	s.cache.set(req.RequestId, resp)
	log.Printf("成功获取异步结果: %s\n", req.RequestId)
	return resp, nil
}

// 批量内容审核
func (s *contentModerationServer) ModerateBatchContent(ctx context.Context, req *pb.BatchModerationRequest) (*pb.BatchModerationResponse, error) {
	log.Printf("收到批量审核请求: %d条内容\n", len(req.Requests))

	// 设置较长的RPC超时
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	// 转发到NLP服务
	resp, err := s.nlpClient.ModerateBatchContent(ctx, req)
	if err != nil {
		log.Printf("NLP服务批量调用失败: %v\n", err)
		return nil, status.Errorf(codes.Internal, "内部服务调用失败: %v", err)
	}

	log.Printf("批量审核完成: %d条内容\n", len(resp.Responses))
	return resp, nil
}

func main() {
	// 解析命令行参数
	flag.Parse()

	// 从环境变量覆盖配置
	if envPort := os.Getenv("SERVICE_PORT"); envPort != "" {
		fmt.Sscanf(envPort, "%d", port)
	}
	if envNLPAddr := os.Getenv("NLP_SERVICE_ADDR"); envNLPAddr != "" {
		*nlpAddress = envNLPAddr
	}

	log.Printf("启动RPC服务器，端口: %d, NLP服务地址: %s\n", *port, *nlpAddress)

	// 创建监听器
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("监听端口失败: %v", err)
	}

	// 连接到NLP服务
	nlpConn, err := grpc.Dial(*nlpAddress, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("连接NLP服务失败: %v", err)
	}
	defer nlpConn.Close()
	nlpClient := pb.NewContentModerationServiceClient(nlpConn)

	// 创建gRPC服务器
	s := grpc.NewServer()
	pb.RegisterContentModerationServiceServer(s, &contentModerationServer{
		nlpClient: nlpClient,
		cache:     newResultCache(),
	})

	// 注册反射服务
	reflection.Register(s)
	log.Printf("已启用gRPC反射服务")

	log.Printf("RPC服务器准备就绪，监听端口: %d", *port)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("服务启动失败: %v", err)
	}
}
