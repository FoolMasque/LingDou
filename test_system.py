# test_system.py - 测试脚本
"""
系统测试脚本
"""
import asyncio
import json
from pathlib import Path


async def test_furniture_processing():
    """测试家具数据处理"""

    # 准备测试数据
    test_data = [
        {
            "风格": "侘寂风系列",
            "子类": "木与石系列",
            "商品名": "木与石床（CJ-E035）",
            "subtitle": "材质：榆木+石头\n尺寸：床屏尺寸：320*110cm/适用床垫尺寸：180*200cm（长*宽*高）\n侘寂家具 OEM 工厂 自有品牌 支持定制",
            "keyword": "大床、石头、实木床、双人床",
            "cover_pic": "https://hjoss.mosyy.com/uploads/mall15991/20250724/8047850/fe1ec508cfd17d1d6f47ef247978f60e.jpg",
            "detail_images": [
                "https://hjoss.mosyy.com/uploads/mall15991/20250724/8047850/d259ea2e74c74eb93c90f1d035db02d5.jpg"
            ]
        }
    ]

    # 保存测试数据到文件
    test_file = Path("test_furniture_data.json")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    # 初始化系统
    from core.system import ProductionCoreSystem
    from core.components import BusinessConfig

    system = ProductionCoreSystem()

    # 注册家具业务
    furniture_config = BusinessConfig(
        business_id="furniture_test",
        name="侘界家具",
        image_fields=["cover_pic", "detail_images"],
        text_fields=["商品名", "子类", "风格", "subtitle", "keyword"]
    )

    system.register_business(furniture_config)

    # 处理数据
    print("开始处理数据...")
    result = await system.process_crawler_data("furniture_test", str(test_file))
    print(f"处理结果: {result}")

    # 测试查询
    print("\n测试查询...")
    queries = [
        "推荐一个实木床",
        "有什么侘寂风格的家具",
        "木与石系列的产品",
    ]

    for query in queries:
        print(f"\n查询: {query}")
        response = await system.query("furniture_test", query)
        # print(f"回答: {response[:200]}...")  # 只打印前200字符
        print(f"回答: {response}...")

    # 检查图片映射
    print("\n检查图片映射...")
    from utils.url_helper import path_manager

    print(f"本地到远程映射数量: {len(path_manager.local_to_remote)}")
    for local, remote in list(path_manager.local_to_remote.items())[:3]:
        print(f"  {local} -> {remote}")


# 运行测试
if __name__ == "__main__":
    import os

    # 设置环境变量
    os.environ['API_KEY'] = "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"
    os.environ['LLM_PROVIDER'] = "openai"

    # 运行异步测试
    asyncio.run(test_furniture_processing())

# 使用curl测试API的示例
"""
# 1. 处理数据
curl -X POST "http://localhost:8008/api/process_data" \
  -H "Content-Type: application/json" \
  -d '{
    "business_id": "furniture_test",
    "json_file": "furniture_data.json"
  }'

# 2. 查询
curl -X POST "http://localhost:8008/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "business_id": "furniture_test",
    "query": "推荐一个茶桌",
    "mode": "hybrid"
  }'

# 3. 检查状态
curl "http://localhost:8008/api/status/furniture_test"

# 4. 访问图片
# 处理后的图片可以通过以下URL访问：
# http://localhost:8008/images/furniture_test/9398025483e0.jpg
"""