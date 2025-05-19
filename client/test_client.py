from client import ModerationClient

def main():
    # 创建客户端
    client = ModerationClient()
    
    try:
        # 测试正常内容
        print("\n测试正常内容:")
        result = client.moderate_content(
            content="这是一个正常的测试内容",
            content_id="test_1"
        )
        print(f"审核结果: {result}")
        
        # 测试敏感内容
        print("\n测试敏感内容:")
        result = client.moderate_content(
            content="这是一个包含敏感词的测试内容",
            content_id="test_2"
        )
        print(f"审核结果: {result}")
        
        # 测试语义侮辱内容
        print("\n测试语义侮辱内容:")
        result = client.moderate_content(
            content="傻逼",
            content_id="test_3"
        )
        print(f"审核结果: {result}")
        
        # 测试异步审核
        print("\n测试异步审核:")
        request_id = client.moderate_content_async(
            content="这是异步审核的测试内容",
            content_id="test_4"
        )
        print(f"异步请求ID: {request_id}")
        
        # 等待结果
        import time
        time.sleep(2)
        
        # 获取结果
        result = client.get_moderation_result(request_id)
        print(f"异步审核结果: {result}")
        
    finally:
        client.close()

if __name__ == "__main__":
    main() 