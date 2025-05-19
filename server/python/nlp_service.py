import os
import sys
import grpc
import time
import uuid
import logging
from concurrent import futures
from typing import Dict, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# 导入NLP引擎
from server.python.nlp_engine import NLPModerationEngine

# 生成protobuf代码
import subprocess
os.chdir(project_root)
proto_dir = os.path.join(project_root, "proto")
proto_file = os.path.join(proto_dir, "contentmoderation.proto")

# 确保proto目录存在
if not os.path.exists(proto_dir):
    os.makedirs(proto_dir)

# 生成Python代码
python_out = os.path.join(project_root, "server", "python")
subprocess.run(["python", "-m", "grpc_tools.protoc", 
                f"--proto_path={proto_dir}", 
                f"--python_out={python_out}", 
                f"--grpc_python_out={python_out}", 
                proto_file])

# 导入生成的protobuf模块
sys.path.append(python_out)
import contentmoderation_pb2 as pb
import contentmoderation_pb2_grpc as pb_grpc

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 存储异步请求结果
async_results = {}

class ContentModerationServicer(pb_grpc.ContentModerationServiceServicer):
    """内容审核服务实现"""
    
    def __init__(self):
        """初始化服务"""
        self.engine = NLPModerationEngine(
            model_dir=os.path.join(project_root, "models"),
            rules_file=os.path.join(project_root, "rules", "rules.json")
        )
        logger.info("内容审核服务初始化完成")
    
    def _create_category_result(self, result):
        """创建类别结果Proto对象"""
        category_results = []
        for cat_result in result["category_results"]:
            category_results.append(pb.CategoryResult(
                category=cat_result["category"],
                detected=cat_result["detected"],
                confidence=cat_result["confidence"],
                matched_keywords=cat_result["matched_keywords"]
            ))
        return category_results
    
    def ModerateContent(self, request, context):
        """同步内容审核"""
        logger.info(f"收到同步审核请求: content_id={request.content_id}")
        
        try:
            # 调用NLP引擎进行内容审核
            result = self.engine.predict(
                text=request.content,
                categories=list(request.categories)
            )
            
            # 构建响应
            category_results = self._create_category_result(result)
            
            response = pb.ModerationResponse(
                content_id=request.content_id,
                is_approved=result["is_approved"],
                category_results=category_results,
                rejection_reason=result["rejection_reason"],
                confidence_score=result["confidence_score"],
                metadata=request.metadata
            )
            
            logger.info(f"同步审核完成: content_id={request.content_id}, is_approved={result['is_approved']}")
            return response
            
        except Exception as e:
            logger.error(f"审核过程出错: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"内部错误: {str(e)}")
            return pb.ModerationResponse()
    
    def ModerateContentAsync(self, request, context):
        """异步内容审核"""
        request_id = str(uuid.uuid4())
        logger.info(f"收到异步审核请求: request_id={request_id}, content_id={request.content_id}")
        
        # 创建异步任务
        def process_async():
            try:
                result = self.engine.predict(
                    text=request.content,
                    categories=list(request.categories)
                )
                
                # 构建响应并存储
                category_results = self._create_category_result(result)
                
                response = pb.ModerationResponse(
                    content_id=request.content_id,
                    is_approved=result["is_approved"],
                    category_results=category_results,
                    rejection_reason=result["rejection_reason"],
                    confidence_score=result["confidence_score"],
                    metadata=request.metadata
                )
                
                async_results[request_id] = {
                    "status": "completed",
                    "response": response
                }
                logger.info(f"异步审核完成: request_id={request_id}, is_approved={result['is_approved']}")
                
            except Exception as e:
                logger.error(f"异步审核出错: request_id={request_id}, error={e}")
                async_results[request_id] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # 标记为处理中
        async_results[request_id] = {"status": "processing"}
        
        # 启动异步处理线程
        executor = futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(process_async)
        
        return pb.AsyncModerationResponse(
            request_id=request_id,
            status="processing"
        )
    
    def GetModerationResult(self, request, context):
        """获取异步审核结果"""
        request_id = request.request_id
        logger.info(f"获取异步审核结果: request_id={request_id}")
        
        if request_id not in async_results:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"未找到请求ID: {request_id}")
            return pb.ModerationResponse()
        
        result = async_results[request_id]
        status = result["status"]
        
        if status == "processing":
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("处理中，请稍后再试")
            return pb.ModerationResponse()
        elif status == "error":
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"处理出错: {result['error']}")
            return pb.ModerationResponse()
        else:
            # 返回结果并从字典中删除
            response = result["response"]
            # 异步结果获取后保留一段时间，避免重复请求
            # 实际生产环境中可能需要更复杂的结果管理机制
            # del async_results[request_id]
            return response
    
    def ModerateBatchContent(self, request, context):
        """批量内容审核"""
        batch_size = len(request.requests)
        logger.info(f"收到批量审核请求: 数量={batch_size}")
        
        responses = []
        for req in request.requests:
            try:
                # 调用NLP引擎进行内容审核
                result = self.engine.predict(
                    text=req.content,
                    categories=list(req.categories)
                )
                
                # 构建响应
                category_results = self._create_category_result(result)
                
                response = pb.ModerationResponse(
                    content_id=req.content_id,
                    is_approved=result["is_approved"],
                    category_results=category_results,
                    rejection_reason=result["rejection_reason"],
                    confidence_score=result["confidence_score"],
                    metadata=req.metadata
                )
                
                responses.append(response)
                
            except Exception as e:
                logger.error(f"批量审核单项出错: content_id={req.content_id}, error={e}")
                # 出错时添加一个空响应
                responses.append(pb.ModerationResponse(
                    content_id=req.content_id,
                    is_approved=False,
                    rejection_reason=f"处理出错: {str(e)}"
                ))
        
        logger.info(f"批量审核完成: 数量={len(responses)}")
        return pb.BatchModerationResponse(responses=responses)


def serve():
    """启动RPC服务器"""
    # 创建gRPC服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_ContentModerationServiceServicer_to_server(
        ContentModerationServicer(), server
    )
    
    # 配置服务器端口
    port = os.environ.get("SERVICE_PORT", "50052")
    server.add_insecure_port(f"[::]:{port}")
    
    # 启动服务器
    server.start()
    logger.info(f"NLP审核服务已启动，监听端口: {port}")
    
    try:
        # 保持服务器运行
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
        logger.info("服务器已关闭")


if __name__ == "__main__":
    serve() 