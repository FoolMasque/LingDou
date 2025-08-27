"""
Example of directly using modal processors

This example demonstrates how to use oldRAG-Anything's modal processors directly without going through MinerU.
"""

import asyncio
import argparse
#rom lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.llm.zhipu import zhipu_complete_if_cache, zhipu_embedding
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag import LightRAG
from raganything import RAGAnything
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
)
import json
import re  # 如果你要用正则 fallback 解析 JSON 的话



WORKING_DIR = "./rag_storage_test_3"


def get_llm_model_func(api_key: str, base_url: str = None):
    return (
        lambda prompt,
        system_prompt=None,
        history_messages=[],
        **kwargs: zhipu_complete_if_cache(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            model="glm-4.5v", #"glm-4.1v-thinking-flash",
            api_key=api_key,
            **kwargs
        )
    )


def get_vision_model_func(api_key: str, base_url: str = None):
    return (
        lambda prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        **kwargs: zhipu_complete_if_cache(
            prompt=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}} if image_data else {}
            ] if image_data else prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            model="glm-4.5v", #"glm-4.1v-thinking-flash",
            api_key=api_key,
            **kwargs
        )
        if image_data
        else zhipu_complete_if_cache(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            model="glm-4.5v", #"glm-4.1v-thinking-flash",
            api_key=api_key,
            **kwargs
        )
    )


def generate_product_caption(product_json: dict) -> str:
    # fields = [
    #     f"- {key}: {value}" for key, value in product_json.items()
    #     if key in ["风格", "子类", "商品名", "subtitle", "keyword"] and value
    # ]
    # fields_str = "\n".join(fields)

    # 定义需要排除的字段
    excluded_fields = {"detail_images", "cover_pic"}
    
    # 动态提取所有非排除字段，并过滤空值
    fields = [
        f"- {key}: {value}" for key, value in product_json.items()
        if key not in excluded_fields and value
    ]
    
    # 将字段拼接为字符串
    fields_str = "\n".join(fields) if fields else "无可用字段信息"
    # caption = (
    #     #f"请用中文详细分析图像中的商品信息，并结合以下提供的产品信息生成结构化 JSON 输出：\n"
    #     f"{fields_str}\n"
    #     #"请严格提取图像中的文字内容（不要省略），并分析商品的视觉呈现（材质、颜色、设计风格、使用场景等）。返回 JSON 格式如下：\n"
    #     "以下是该商品的已知信息（仅供缺失字段回填，禁止臆造）：\n"
    #     "输出必须为**单一 JSON 对象**，字段：style, category, name, subtitle, keyword, description。\n"
    #     "description ≤50字，仅描述商品本体材质/颜色/形态/功能，禁止出现背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯等词。"
    # )
    caption = (
        "以下是该商品的已知信息（仅供缺失字段回填，禁止臆造）：\n"
        f"{fields_str}\n"
    )
    return caption


async def process_image_example(lightrag: LightRAG, vision_model_func):
    """Example of processing an image"""

    """
    1) 用 RAGAnything 解析图片并写入 LightRAG 知识库
    2) 解析完成后，立刻调用 rag.aquery 进行查询（支持 hybrid/local/global/naive）
    """
    # —— 关键：基于已有的 LightRAG 实例，创建 raganything 包装器
    rag = RAGAnything(lightrag=lightrag)

    # 可选：确保底层存储初始化（有些版本需要手动调一次）
    await lightrag.initialize_storages()


    # Create image processor
    image_processor = ImageModalProcessor(
        lightrag=lightrag, modal_caption_func=vision_model_func
    )

 

    # 文件路径
    file_path_1 = "C:\\Users\\81502\\Desktop\\家具数据\\1mosyy_Enterprise_simplicity.json"
    file_path_2 = "C:\\Users\\81502\\Desktop\\家具数据\\2mosyy_goods_test.json"
    file_path_3 = "C:\\Users\\81502\\Desktop\\家具数据\\3mosyy_cases_test.json"
    file_path_33 = "C:\\Users\\81502\\Desktop\\家具数据\\3mosyy_cases_catalog_pic_urls.json"

    # # 打开并读取文件
    # with open(file_path_2, 'r', encoding='utf-8') as file:
    #     try:
    #         # 加载JSON数据
    #         product_jsons = json.load(file)
            
    #         # 逐个处理每个产品对象
    #         for product_index, product_json in enumerate(product_jsons):
    #             # 动态生成 img_caption
    #             product_caption = generate_product_caption(product_json)

    #             # 准备 image_content
    #             image_content = {
    #                 "img_path": {
    #                     "cover_pic": product_json.get("cover_pic", ""),
    #                     "detail_images": product_json.get("detail_images", [])
    #                 },
    #                 "img_caption": [product_caption],
    #                 "img_footnote": []
    #             }

    #             # 处理图像（假设是异步函数，使用 await）
    #             (description, entity_info, image_info_blocks) = await image_processor.process_multimodal_content(
    #                 modal_content=image_content,
    #                 content_type="image",
    #                 file_path=product_json.get("cover_pic", ""),  
    #                 #file_path=f"product_{product_index}_image.jpg",
    #                 entity_name=f"{product_json['商品名']} 图像分析"
    #             )
    #     except json.JSONDecodeError as e:
    #         print(f"JSON解析错误: {e}")


    # # 打开并读取文件
    # with open(file_path_3, 'r', encoding='utf-8') as file:
    #     try:
    #         # 加载JSON数据
    #         product_jsons = json.load(file)
            
    #         # 逐个处理每个产品对象
    #         for product_index, product_json in enumerate(product_jsons):
    #             # 动态生成 img_caption
    #             product_caption = generate_product_caption(product_json)

    #             # 准备 image_content
    #             image_content = {
    #                 "img_path": {
    #                     "cover_pic": product_json.get("cover_pic", ""),
    #                     "detail_images": product_json.get("detail_images", [])
    #                 },
    #                 "img_caption": [product_caption],
    #                 "img_footnote": []
    #             }
    #             # 处理图像（假设是异步函数，使用 await）
    #             (description, entity_info, image_info_blocks) = await image_processor.process_multimodal_content(
    #                 modal_content=image_content,
    #                 content_type="image",
    #                 file_path=product_json.get("detail_images", "")[0],  
    #                 entity_name=f"{product_json['title']} 图像分析"
    #             )
    #     except json.JSONDecodeError as e:
    #         print(f"JSON解析错误: {e}")
    

 

    #===== 2) 4种模式问答 =====
    user_questions = [
        #"我有一块100*80*50的空间，请你给我推荐合适的茶几"
        #"我想要底座是圆的茶几，请你推荐几个"
        # "我想要底座是圆的边几，请推荐几个",
        # "我想要工艺为雕刻的家具，请推荐几个"
        #"我想要底座是圆的边几，请至少推荐5个，并给上对应的封面图片"
        # "我想要工艺为雕刻的家具，请至少推荐10个"
        # "我想要工艺为雕刻并且底座为圆形的家具，请至少推荐5个，并给上对应的封面图片链接"
        #"我想要一个镶嵌碳炉的桌子，请你给我推荐几个，并给上对应的封面图片链接",
        # "各商品材质的差异是什么？请按商品名逐条列出，并给出来源。",
        # "使用青石材质的家具有哪些",
        # "侘寂风格是什么"
        # "我想要防水的沙发，请给我推荐几个，并给上对应的封面图片链接",
        # "我想要风化木的家具，请给我推荐几个"\
        "我想要了解一下广西的案例"
    ]

    for q in user_questions:
        print("\n================= 查询 =================")
        print("Q:", q)

        # （1）"hybrid": Combines local and global retrieval methods.
        res_hybrid = await rag.aquery(q, mode="hybrid")
        print("\n[hybrid]\n", res_hybrid)

        # （2）"local": Focuses on context-dependent information.
        res_local = await rag.aquery(q, mode="local")
        print("\n[local]\n", res_local)

        # （3）"global": Utilizes global knowledge.
        res_global = await rag.aquery(q, mode="global")
        print("\n[global]\n", res_global)

        # （4）"naive": Performs a basic search without advanced techniques.
        res_naive = await rag.aquery(q, mode="naive")
        print("\n[naive]\n", res_naive)

    # Prepare image content
    #image_content = {
        # "img_path": "C:/Users/81502/Desktop/test_rag/f51e98a666cd81e42beaf082f026612e.jpg",  
        # "img_path": "https://hjoss.mosyy.com/uploads/mall15991/20241106/f51e98a666cd81e42beaf082f026612e.jpg",
        # "img_path": [
        #     "https://hjoss.mosyy.com/uploads/mall15991/20241106/f51e98a666cd81e42beaf082f026612e.jpg",
        #     "https://hjoss.mosyy.com/uploads/mall15991/20241106/d6821bef515c345524a4cc180bc0133c.jpg",
        #     "https://hjoss.mosyy.com/uploads/mall15991/20241106/9beb3e8b30767be8a7d73866e5eab628.jpg",
        #     "https://hjoss.mosyy.com/uploads/mall15991/20241106/8b62bfb870bf3105493c7d8b2f084e48.jpg"
        # ],         ##3空间案例目录
        
        #"img_path": [
            # "https://hjoss.mosyy.com/uploads/mall15991/20250516/9350d83e41f9466952c6b7f327ce6bf9.jpg",
            # "https://hjoss.mosyy.com/uploads/mall15991/20241105/cfc9a7cc1e369c1f622655a938cce922.jpg",
            # "https://hjoss.mosyy.com/uploads/mall15991/20241105/d4a27cd1d2fb3e53490b64f7d911908d.jpg",
            # "https://hjoss.mosyy.com/uploads/mall15991/20241105/a192f8841f8ca6f42b9823cf756c5ad1.jpg",
            # "https://hjoss.mosyy.com/uploads/mall15991/20241105/f60c61262bd3e6e89d17b868934e7f73.jpg",
            # "https://hjoss.mosyy.com/uploads/mall15991/20241105/b361145abc55be57e70f39ef85ff4a8a.jpg"             
            # 企业简介
        # ],
        
        #"img_caption": ["请详细分析该图像中展示的空间设计案例，根据图片信息提取出每个空间案例的名字，并指出其所属空间类型（如酒店、度假村、商业空间等）、设计风格（如侘寂、现代、极简、中式等）、主要使用人群、使用场景、设计亮点与特色材料。"],  

        ###3 空间案例目录
        #"img_caption": ["请用中文回答！！请详细分析该图像中的空间设计案例：1. 明确提取每个面板上的标题、项目名称与地点（比如1.浙江绍兴大乐之野）；2. 判断每个空间的类型（如酒店、度假村、商业空间等）；3. 识别每个空间的设计风格（如侘寂、现代、极简、中式等）；4. 推断主要使用人群与使用场景；5. 指出每个空间的设计亮点与使用的特色材料。请逐条回答，每一条对应一个面板的分析。"], 


        ###1 企业简介 
        #"img_caption": ["请用中文严格提取其中的文字内容，不要省略"],
        #"img_caption": ["请用中文严格提取其中的文字内容，不要省略并用一句话整体描述侧重于图片上和商品名称相关的内容"],
        #"img_footnote": [""]  
    #}

    

    # # Process image
    # (description, entity_info,_) = await image_processor.process_multimodal_content(
    #     modal_content=image_content,
    #     content_type="image",
    #     file_path="image_example.jpg",
    #     entity_name="Example Image"
    # )

    # print("\n📷 Image Processing Results:")
    # print(f"\n📝 Description:\n{description}\n")
    # print(f"📦 Entity Info:\n{json.dumps(entity_info, indent=2, ensure_ascii=False)}")



async def process_table_example(lightrag: LightRAG, llm_model_func):
    """Example of processing a table"""
    # Create table processor
    table_processor = TableModalProcessor(
        lightrag=lightrag, modal_caption_func=llm_model_func
    )

    # Prepare table content
    table_content = {
        "table_body": """
        | Name | Age | Occupation |
        |------|-----|------------|
        | John | 25  | Engineer   |
        | Mary | 30  | Designer   |
        """,
        "table_caption": ["Employee Information Table"],
        "table_footnote": ["Data updated as of 2024"],
    }

    # Process table
    (description, entity_info, _) = await table_processor.process_multimodal_content(
        modal_content=table_content,
        content_type="table",
        file_path="table_example.md",
        entity_name="Employee Table",
    )

    print("\nTable Processing Results:")
    print(f"Description: {description}")
    print(f"Entity Info: {entity_info}")


async def process_equation_example(lightrag: LightRAG, llm_model_func):
    """Example of processing a mathematical equation"""
    # Create equation processor
    equation_processor = EquationModalProcessor(
        lightrag=lightrag, modal_caption_func=llm_model_func
    )

    # Prepare equation content
    equation_content = {"text": "E = mc^2", "text_format": "LaTeX"}

    # Process equation
    (description, entity_info, _) = await equation_processor.process_multimodal_content(
        modal_content=equation_content,
        content_type="equation",
        file_path="equation_example.txt",
        entity_name="Mass-Energy Equivalence",
    )

    print("\nEquation Processing Results:")
    print(f"Description: {description}")
    print(f"Entity Info: {entity_info}")


async def initialize_rag(api_key: str, base_url: str = None):
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            # embedding_dim=3072,
            embedding_dim=2048,
            max_token_size=8192,
            func=lambda texts: zhipu_embedding(
                texts,
                model="embedding-3",
                api_key=api_key
            ),
        ),
        llm_model_func=lambda prompt,
        system_prompt=None,
        history_messages=[],
        **kwargs: zhipu_complete_if_cache(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            model="glm-4.5v", #"glm-4.1v-thinking-flash",
            api_key=api_key,
            **kwargs
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="Modal Processors Example")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", help="Optional base URL for API")
    parser.add_argument(
        "--working-dir", "-w", default=WORKING_DIR, help="Working directory path"
    )

    args = parser.parse_args()

    # Run examples
    asyncio.run(main_async(args.api_key, args.base_url))


async def main_async(api_key: str, base_url: str = None):
    # Initialize LightRAG
    lightrag = await initialize_rag(api_key, base_url)

    # Get model functions
    llm_model_func = get_llm_model_func(api_key, base_url)
    vision_model_func = get_vision_model_func(api_key, base_url)

    # Run examples
    await process_image_example(lightrag, vision_model_func)
    # await process_table_example(lightrag, llm_model_func)
    # await process_equation_example(lightrag, llm_model_func)


if __name__ == "__main__":
    # main()
    asyncio.run(main_async(api_key="7d5e548df35742879a74227105b367e8.vsjrDZodWf2oIppH" ))  # Replace with your actual API key

    # import base64

    # with open("C:/Users/81502/Desktop/test_rag/f51e98a666cd81e42beaf082f026612e.jpg", "rb") as f:
    #     b64_local = base64.b64encode(f.read()).decode("utf-8")
    #     print("[本地图片] base64 长度：", len(b64_local))
    #     print("[本地图片] 前100字符：", b64_local[:100])

    # import base64
    # import requests

    # url = "https://hjoss.mosyy.com/uploads/mall15991/20241106/f51e98a666cd81e42beaf082f026612e.jpg"
    # response = requests.get(url)
    # response.raise_for_status()
    # b64_remote = base64.b64encode(response.content).decode("utf-8")
    # print("[远程图片] base64 长度：", len(b64_remote))
    # print("[远程图片] 前100字符：", b64_remote[:100])


#一样的
# [本地图片] base64 长度： 757648
# [本地图片] 前100字符： /9j/4QCmRXhpZgAASUkqAAgAAAADADEBAgAiAAAAMgAAADIBAgAaAAAAVAAAAGmHBAABAAAAbgAAAAAAAABBZG9iZSBQaG90b3No
# [远程图片] base64 长度： 757648
# [远程图片] 前100字符： /9j/4QCmRXhpZgAASUkqAAgAAAADADEBAgAiAAAAMgAAADIBAgAaAAAAVAAAAGmHBAABAAAAbgAAAAAAAABBZG9iZSBQaG90b3No