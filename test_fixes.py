#!/usr/bin/env python3
"""
修复验证测试脚本
"""
import json
import asyncio
from pathlib import Path
from config.settings import settings
from core.components import MultiModalProcessor, ImageManager
from core.rag_instance import ProductionRAGInstance
from utils.url_helper import path_manager, debug_image_access
from utils.logger import setup_logger

logger = setup_logger(__name__)


async def test_image_processing():
    """测试图片处理功能"""
    print("=== 测试图片处理功能 ===")

    # 测试数据
    test_item = {
        "风格": "侘寂风系列",
        "子类": "木与石系列",
        "商品名": "木与石床（CJ-E035）",
        "subtitle": "材质：榆木+石头\n尺寸：床屏尺寸：320*110cm/适用床垫尺寸：180*200cm（长*宽*高）",
        "keyword": "大床、石头、实木床、双人床",
        "cover_pic": "https://hjoss.mosyy.com/uploads/mall15991/20250724/8047850/fe1ec508cfd17d1d6f47ef247978f60e.jpg",
        "detail_images": [
            "https://hjoss.mosyy.com/uploads/mall15991/20250724/8047850/fe1ec508cfd17d1d6f47ef247978f60e.jpg",
            "https://hjoss.mosyy.com/uploads/mall15991/20250724/8047850/d259ea2e74c74eb93c90f1d035db02d5.jpg"
        ]
    }

    # 1. 测试多模态处理器
    print("\n1. 测试多模态处理器...")
    processor = MultiModalProcessor("furniture")
    modal_content = processor.build_modal_content(test_item)

    print(f"生成的模态内容:")
    print(f"  图片路径数量: {len(modal_content.get('img_path', {}))}")
    for key, path in modal_content.get('img_path', {}).items():
        print(f"    {key}: {path}")

    # 2. 测试图片管理器
    print("\n2. 测试图片管理器...")
    image_manager = ImageManager()

    # 提取图片URL
    image_urls = []
    if test_item.get("cover_pic"):
        image_urls.append(test_item["cover_pic"])
    if test_item.get("detail_images"):
        image_urls.extend(test_item["detail_images"])

    print(f"需要下载的图片: {len(image_urls)}")

    # 下载图片（测试环境可以注释掉实际下载）
    # image_mappings = await image_manager.download_images(image_urls, "furniture")
    # print(f"下载完成: {len(image_mappings)}")

    return True


async def test_prompts():
    """测试提示词系统"""
    print("\n=== 测试提示词系统 ===")

    from config.prompts import ChinesePrompts

    # 测试提示词获取
    print("1. 图像分析提示词:")
    image_prompt = ChinesePrompts.get_image_analysis_prompt()
    print(f"  长度: {len(image_prompt)}")
    print(f"  前100字符: {image_prompt[:100]}...")

    print("\n2. 实体抽取提示词:")
    entity_prompt = ChinesePrompts.get_entity_extraction_prompt()
    print(f"  长度: {len(entity_prompt)}")
    print(f"  前100字符: {entity_prompt[:100]}...")

    print("\n3. 查询响应提示词:")
    query_prompt = ChinesePrompts.get_furniture_query_prompt("推荐一个茶桌")
    print(f"  长度: {len(query_prompt)}")
    print(f"  前100字符: {query_prompt[:100]}...")

    # 测试版本信息
    version_info = ChinesePrompts.get_version_info()
    print(f"\n4. 提示词版本信息: {version_info}")

    return True


def test_url_processing():
    """测试URL处理"""
    print("\n=== 测试URL处理功能 ===")

    from utils.url_helper import post_process_response_urls, path_manager

    # 注册测试映射
    path_manager.register_mapping(
        "../static/images/furniture/test123.jpg",
        "http://localhost:8008/images/furniture/test123.jpg"
    )
    # 调试：检查映射注册的结果
    print(f"调试：注册的映射结果: {path_manager.get_remote_url('../static/images/furniture/test123.jpg')}")

    # 测试URL替换
    test_texts = [
        "产品图片：../static/images/furniture/test123.jpg",
        "查看详情：static/images/furniture/test123.jpg",
        "图片路径：./images/furniture/test123.jpg",
        "Windows路径：..\\static\\images\\furniture\\70f30a920a03.jpg",
        "混合内容：这是产品 ../static/images/furniture/test123.jpg 的介绍"
    ]

    print("URL处理测试:")
    for i, text in enumerate(test_texts, 1):
        processed = post_process_response_urls(text)
        print(f"  {i}. 原文: {text}")
        print(f"     处理: {processed}")
        print()

    return True


async def test_rag_instance():
    """测试RAG实例"""
    print("\n=== 测试RAG实例 ===")

    try:
        # 创建RAG实例（不初始化，只测试创建过程）
        rag = ProductionRAGInstance("furniture_test")
        print(f"RAG实例创建成功: {rag.business_id}")
        print(f"工作目录: {rag.working_dir}")
        print(f"已初始化: {rag.initialized}")

        # 测试中文LLM函数获取
        llm_func = rag._get_chinese_llm_func()
        print(f"中文LLM函数创建成功: {callable(llm_func)}")

        return True

    except Exception as e:
        print(f"RAG实例测试失败: {e}")
        return False


def test_configuration():
    """测试配置系统"""
    print("\n=== 测试配置系统 ===")

    print(f"配置信息:")
    print(f"  Provider: {settings.provider}")
    print(f"  LLM Model: {settings.llm_model}")
    print(f"  Vision Model: {settings.vision_model}")
    print(f"  Embedding Model: {settings.embedding_model}")
    print(f"  中文提示词: {settings.use_chinese_prompts}")
    print(f"  静态URL基础: {settings.static_base_url}")
    print(f"  图片存储目录: {settings.image_storage}")
    print(f"  端口: {settings.port}")

    # 检查关键目录
    from pathlib import Path

    working_dir = Path(settings.working_dir + "_furniture")
    image_dir = Path(settings.image_storage)

    print(f"\n目录状态:")
    print(f"  工作目录存在: {working_dir.exists()}")
    print(f"  图片目录存在: {image_dir.exists()}")

    if image_dir.exists():
        furniture_dir = image_dir / "furniture"
        print(f"  家具图片目录存在: {furniture_dir.exists()}")
        if furniture_dir.exists():
            image_count = len(list(furniture_dir.glob("*.jpg"))) + len(list(furniture_dir.glob("*.png")))
            print(f"  现有图片数量: {image_count}")

    return True


async def main():
    """主测试函数"""
    print("=== LingDou RAG系统修复验证测试 ===")
    print(f"测试时间: {asyncio.get_event_loop().time()}")

    tests = [
        ("配置系统", test_configuration),
        ("图片处理", test_image_processing),
        ("提示词系统", test_prompts),
        ("URL处理", test_url_processing),
        ("RAG实例", test_rag_instance),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'=' * 50}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
            print(f"✓ {test_name} 测试{'通过' if result else '失败'}")
        except Exception as e:
            results[test_name] = False
            print(f"✗ {test_name} 测试异常: {e}")

    # 输出测试总结
    print(f"\n{'=' * 50}")
    print("测试总结:")
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    print(f"通过: {passed}/{total}")

    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")

    # 输出修复建议
    if passed < total:
        print(f"\n修复建议:")
        for test_name, result in results.items():
            if not result:
                print(f"  - 检查 {test_name} 相关配置和依赖")

    print(f"\n测试完成!")
    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
        exit(0 if all(results.values()) else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        exit(1)
    except Exception as e:
        print(f"\n测试执行异常: {e}")
        exit(1)