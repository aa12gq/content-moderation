import os
import sys
import grpc
import time
import logging
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# 导入生成的protobuf模块
import subprocess
os.chdir(project_root)
proto_dir = os.path.join(project_root, "proto")
proto_file = os.path.join(proto_dir, "contentmoderation.proto")

# 确保proto目录存在
if not os.path.exists(proto_dir):
    raise RuntimeError(f"找不到proto目录: {proto_dir}")

# 生成Python代码
client_out = os.path.join(project_root, "client")
subprocess.run(["python", "-m", "grpc_tools.protoc", 
                f"--proto_path={proto_dir}", 
                f"--python_out={client_out}", 
                f"--grpc_python_out={client_out}", 
                proto_file])

# 导入生成的protobuf模块
sys.path.append(client_out)
import contentmoderation_pb2 as pb
import contentmoderation_pb2_grpc as pb_grpc

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModerationClient:
    """内容审核客户端"""
    
    def __init__(self, server_address: str = "localhost:50051"):
        """
        初始化客户端
        
        Args:
            server_address: 服务器地址，格式为"host:port"
        """
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = pb_grpc.ContentModerationServiceStub(self.channel)
        logger.info(f"已连接到内容审核服务: {server_address}")
    
    def close(self):
        """关闭连接"""
        self.channel.close()
        logger.info("已关闭连接")
    
    def moderate_content(self, content: str, content_id: str = None, content_type: str = "text", 
                       categories: Optional[List[str]] = None, metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        同步内容审核
        
        Args:
            content: 待审核的内容
            content_id: 内容ID，为空时自动生成
            content_type: 内容类型，默认为text
            categories: 要审核的类别列表，默认为全部类别
            metadata: 元数据
            
        Returns:
            审核结果字典
        """
        if content_id is None:
            content_id = f"content_{int(time.time() * 1000)}"
        
        if categories is None:
            categories = ["spam", "porn", "violence", "political", "insult"]
        
        if metadata is None:
            metadata = {}
        
        # 构建请求
        request = pb.ModerationRequest(
            content_id=content_id,
            content=content,
            content_type=content_type,
            categories=categories,
            metadata=metadata
        )
        
        try:
            # 发送请求
            logger.info(f"发送审核请求: {content_id}")
            response = self.stub.ModerateContent(request)
            
            # 处理响应
            result = {
                "content_id": response.content_id,
                "is_approved": response.is_approved,
                "rejection_reason": response.rejection_reason,
                "confidence_score": response.confidence_score,
                "category_results": []
            }
            
            # 添加类别结果
            for cat in response.category_results:
                result["category_results"].append({
                    "category": cat.category,
                    "detected": cat.detected,
                    "confidence": cat.confidence,
                    "matched_keywords": list(cat.matched_keywords)
                })
            
            logger.info(f"审核结果: content_id={content_id}, is_approved={response.is_approved}")
            return result
            
        except grpc.RpcError as e:
            logger.error(f"RPC错误: {e.code()}: {e.details()}")
            raise
    
    def moderate_content_async(self, content: str, content_id: str = None, content_type: str = "text", 
                             categories: Optional[List[str]] = None, metadata: Optional[Dict[str, str]] = None) -> str:
        """
        异步内容审核
        
        Args:
            content: 待审核的内容
            content_id: 内容ID，为空时自动生成
            content_type: 内容类型，默认为text
            categories: 要审核的类别列表，默认为全部类别
            metadata: 元数据
            
        Returns:
            请求ID
        """
        if content_id is None:
            content_id = f"content_{int(time.time() * 1000)}"
        
        if categories is None:
            categories = ["spam", "porn", "violence", "political", "insult"]
        
        if metadata is None:
            metadata = {}
        
        # 构建请求
        request = pb.ModerationRequest(
            content_id=content_id,
            content=content,
            content_type=content_type,
            categories=categories,
            metadata=metadata
        )
        
        try:
            # 发送请求
            logger.info(f"发送异步审核请求: {content_id}")
            response = self.stub.ModerateContentAsync(request)
            logger.info(f"异步请求已提交: request_id={response.request_id}")
            return response.request_id
            
        except grpc.RpcError as e:
            logger.error(f"RPC错误: {e.code()}: {e.details()}")
            raise
    
    def get_moderation_result(self, request_id: str) -> Dict[str, Any]:
        """
        获取异步审核结果
        
        Args:
            request_id: 请求ID
            
        Returns:
            审核结果字典
        """
        request = pb.ResultRequest(request_id=request_id)
        
        try:
            # 发送请求
            logger.info(f"获取异步结果: {request_id}")
            response = self.stub.GetModerationResult(request)
            
            # 处理响应
            result = {
                "content_id": response.content_id,
                "is_approved": response.is_approved,
                "rejection_reason": response.rejection_reason,
                "confidence_score": response.confidence_score,
                "category_results": []
            }
            
            # 添加类别结果
            for cat in response.category_results:
                result["category_results"].append({
                    "category": cat.category,
                    "detected": cat.detected,
                    "confidence": cat.confidence,
                    "matched_keywords": list(cat.matched_keywords)
                })
            
            logger.info(f"异步审核结果: request_id={request_id}, is_approved={response.is_approved}")
            return result
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                logger.info(f"结果尚未准备好: {request_id}")
                return {"status": "processing"}
            elif e.code() == grpc.StatusCode.NOT_FOUND:
                logger.error(f"未找到请求ID: {request_id}")
                return {"status": "not_found"}
            else:
                logger.error(f"RPC错误: {e.code()}: {e.details()}")
                raise
    
    def moderate_batch_content(self, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量内容审核
        
        Args:
            contents: 内容列表，每项应包含content, content_id, content_type, categories, metadata字段
            
        Returns:
            审核结果列表
        """
        requests = []
        for item in contents:
            content_id = item.get("content_id", f"content_{int(time.time() * 1000)}_{len(requests)}")
            content = item.get("content", "")
            content_type = item.get("content_type", "text")
            categories = item.get("categories", ["spam", "porn", "violence", "political", "insult"])
            metadata = item.get("metadata", {})
            
            requests.append(pb.ModerationRequest(
                content_id=content_id,
                content=content,
                content_type=content_type,
                categories=categories,
                metadata=metadata
            ))
        
        batch_request = pb.BatchModerationRequest(requests=requests)
        
        try:
            # 发送请求
            logger.info(f"发送批量审核请求: {len(requests)}条内容")
            response = self.stub.ModerateBatchContent(batch_request)
            
            # 处理响应
            results = []
            for resp in response.responses:
                result = {
                    "content_id": resp.content_id,
                    "is_approved": resp.is_approved,
                    "rejection_reason": resp.rejection_reason,
                    "confidence_score": resp.confidence_score,
                    "category_results": []
                }
                
                # 添加类别结果
                for cat in resp.category_results:
                    result["category_results"].append({
                        "category": cat.category,
                        "detected": cat.detected,
                        "confidence": cat.confidence,
                        "matched_keywords": list(cat.matched_keywords)
                    })
                
                results.append(result)
            
            logger.info(f"批量审核完成: 共{len(results)}条结果")
            return results
            
        except grpc.RpcError as e:
            logger.error(f"RPC错误: {e.code()}: {e.details()}")
            raise


def main():
    """客户端示例"""
    # 服务器地址
    server_address = os.environ.get("SERVER_ADDRESS", "localhost:50051")
    
    # 创建客户端
    client = ModerationClient(server_address)
    
    try:
        # 同步审核
        print("\n=== 同步内容审核示例 ===")
        test_content1 = "这是一条正常的测试消息，不包含任何敏感内容。"
        result1 = client.moderate_content(test_content1)
        print(f"内容: {test_content1}")
        print(f"审核结果: {'通过' if result1['is_approved'] else '拒绝'}")
        if not result1["is_approved"]:
            print(f"拒绝原因: {result1['rejection_reason']}")
            for cat in result1["category_results"]:
                if cat["detected"]:
                    print(f"- {cat['category']}: 置信度 {cat['confidence']:.2f}")
                    if cat["matched_keywords"]:
                        print(f"  匹配关键词: {', '.join(cat['matched_keywords'])}")
        
        # 敏感内容测试
        print("\n=== 敏感内容审核示例 ===")
        test_content2 = "点击领取限时优惠，免费获取会员资格！这是一条广告信息。"
        result2 = client.moderate_content(test_content2)
        print(f"内容: {test_content2}")
        print(f"审核结果: {'通过' if result2['is_approved'] else '拒绝'}")
        if not result2["is_approved"]:
            print(f"拒绝原因: {result2['rejection_reason']}")
            for cat in result2["category_results"]:
                if cat["detected"]:
                    print(f"- {cat['category']}: 置信度 {cat['confidence']:.2f}")
                    if cat["matched_keywords"]:
                        print(f"  匹配关键词: {', '.join(cat['matched_keywords'])}")
        
        # 异步审核
        print("\n=== 异步内容审核示例 ===")
        test_content3 = "这个内容需要异步处理，包含一些可能的敏感词如'暴力'和'色情'。"
        request_id = client.moderate_content_async(test_content3)
        print(f"异步请求已提交: request_id={request_id}")
        
        # 等待结果
        max_retries = 5
        for i in range(max_retries):
            print(f"等待结果中... (尝试 {i+1}/{max_retries})")
            time.sleep(1)  # 等待1秒
            
            result3 = client.get_moderation_result(request_id)
            if result3.get("status") == "processing":
                continue
            
            print(f"内容: {test_content3}")
            print(f"审核结果: {'通过' if result3['is_approved'] else '拒绝'}")
            if not result3["is_approved"]:
                print(f"拒绝原因: {result3['rejection_reason']}")
                for cat in result3["category_results"]:
                    if cat["detected"]:
                        print(f"- {cat['category']}: 置信度 {cat['confidence']:.2f}")
                        if cat["matched_keywords"]:
                            print(f"  匹配关键词: {', '.join(cat['matched_keywords'])}")
            break
        else:
            print("未能在规定时间内获取结果")
        
        # 批量审核
        print("\n=== 批量内容审核示例 ===")
        batch_contents = [
            {"content": "正常内容示例，这条消息应该通过审核。"},
            {"content": "点击领取限时优惠，免费获取会员资格！"},
            {"content": "这个内容包含暴力描述，如杀人和血腥场景。"},
            {"content": "这是一条正常的产品评价，质量不错，推荐购买。"}
        ]
        
        batch_results = client.moderate_batch_content(batch_contents)
        print(f"批量审核结果: 共{len(batch_results)}条")
        
        for i, result in enumerate(batch_results):
            print(f"\n第{i+1}条内容: {batch_contents[i]['content']}")
            print(f"审核结果: {'通过' if result['is_approved'] else '拒绝'}")
            if not result["is_approved"]:
                print(f"拒绝原因: {result['rejection_reason']}")
                for cat in result["category_results"]:
                    if cat["detected"]:
                        print(f"- {cat['category']}: 置信度 {cat['confidence']:.2f}")
                        if cat["matched_keywords"]:
                            print(f"  匹配关键词: {', '.join(cat['matched_keywords'])}")
    
    finally:
        # 关闭连接
        client.close()


if __name__ == "__main__":
    main() 