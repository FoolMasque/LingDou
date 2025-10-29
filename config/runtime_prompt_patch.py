# config/runtime_prompt_patch.py
"""
中文业务提示词 - 基于业务定制
"""
from utils.logger import setup_logger

logger = setup_logger(__name__)


def apply_chinese_prompts_runtime():
    """运行时应用中文提示词"""

    try:
        from raganything.prompt import PROMPTS

        chinese_prompts = {
            "IMAGE_ANALYSIS_SYSTEM": "你是专业的产品分析师。请分析图片并用中文返回JSON格式的结果。",

            "vision_prompt_with_context": """分析这张产品图片，提供JSON格式回复：
        
        {{
            "detailed_description": "产品的中文描述：材质（木材、石材等）、工艺特征、设计风格（侘寂风等）、功能用途、独特特色",
            "entity_info": {{
                "entity_name": "{entity_name}",
                "entity_type": "image",
                "summary": "产品核心特征总结（不超过50字）"
            }}
        }}
        
        Context from surrounding content:
        {context}
        
        Image details:
        - Image Path: {image_path}
        - Captions: {captions}
        - Footnotes: {footnotes}
        
        简洁专业分析即可。""",

            "vision_prompt": """你是“OCR + 商品信息抽取器”。只输出一个 JSON 对象：

        {{
            "detailed_description": "逐字提取图像中出现的所有文字，按行保留换行和标点；不得改写、总结或省略；如果没有文字，填空字符串，若图中有文字，请一字不落提取文字，并用一句话描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词；不少于20字",
            "entity_info": {{
                "entity_name": "{entity_name}",
                "entity_type": "image",
                "summary": "图像内容、其重要性及与周围内容关系的简明摘要（最多100字）"
            }}
        }}
        
        Additional context:
        - Image Path: {image_path}
        - Captions: {captions}
        - Footnotes: {footnotes}
        
        重点提供准确的视觉分析，结合上下文以支持知识检索。""",

            "image_chunk": "图像内容分析：\n图片路径：{image_path}\n说明：{captions}\n补充：{footnotes}\n\n视觉分析：{enhanced_caption}",

            "generic_prompt": """请分析此{content_type}内容并提供以下结构的 JSON 响应:

        {{
            "detailed_description": "对内容的全面分析，包括:
            - 内容结构和组织
            - 关键信息和元素
            - 组件之间的关系
            - 上下文和重要性
            - 与知识检索相关的详细信息
            请务必使用适合 {content_type} 内容的特定术语。",
            "entity_info": {{
                "entity_name": "{entity_name}",
                "entity_type": "{content_type}",
                "summary": "内容目的和要点的简明摘要（最多 100 字）"
            }}
        }}
        
        Content: {content}
        
        专注于提取对知识检索有用的有意义信息。""",
            "generic_prompt_with_context": """请结合上下文分析此 {content_type} 内容并提供以下结构的 JSON 响应:

                {{
                    "detailed_description": "对内容的全面分析，包括:
                    - 内容结构和组织
                    - 关键信息和元素
                    - 与周围内容相关的背景和重要性
                    - 此内容如何与更广泛的讨论相联系或支持
                    - 与知识检索相关的详细信息
                    请务必使用适合 {content_type} 内容的特定术语。",
                    "entity_info": {{
                        "entity_name": "{entity_name}",
                        "entity_type": "{content_type}",
                        "summary": "简明扼要地概括内容的目的、要点以及与周围环境的关系（最多 100 个字）"
                    }}
                }}

                周围内容的背景：
                {context}

                Content: {content}

                专注于提取对知识检索有用的有意义信息。""",
            "QUERY_IMAGE_DESCRIPTION": "请简要描述这张图片的主要内容、关键元素和重要信息。",
            "QUERY_IMAGE_ANALYST_SYSTEM": "您是一位专业的图像分析师，能够准确描述图像内容。",
            "QUERY_GENERIC_ANALYSIS": """请分析以下 {content_type} 类型的内容，并提取其主要信息和关键特征：

                内容：{content_str}

                请简要概括此内容的主要特征和重要信息。""",
            "QUERY_GENERIC_ANALYST_SYSTEM": "您是一位专业的内容分析师，能够准确分析 {content_type} 类型的内容。",
            "QUERY_ENHANCEMENT_SUFFIX": "\n\n请根据用户查询和提供的多模态内容信息，提供全面的答案。",
        }

        # 直接替换
        for key, value in chinese_prompts.items():
            PROMPTS[key] = value

        logger.info("运行时中文提示词应用成功")
        return True

    except Exception as e:
        logger.error(f"运行时提示词替换失败: {e}")
        return False


def verify_chinese_prompts():
    """验证中文提示词是否生效"""
    try:
        from raganything.prompt import PROMPTS

        # 检查关键提示词
        image_system = PROMPTS.get("IMAGE_ANALYSIS_SYSTEM", "")
        vision_context = PROMPTS.get("vision_prompt_with_context", "")

        if "中文" in image_system and "中文" in vision_context:
            logger.info("验证成功：中文提示词已生效")
            return True
        else:
            logger.warning("验证失败：提示词可能仍为英文")
            return False

    except Exception as e:
        logger.error(f"验证失败: {e}")
        return False


def get_business_specific_caption_builder(business_id: str):
    """根据业务ID获取专用的caption构建器"""

    builders = {
        "furniture": _build_furniture_caption,
        "toilet": _build_toilet_caption,
        "electronics": _build_electronics_caption
    }

    return builders.get(business_id, _build_generic_caption)


def _build_furniture_caption(item: dict) -> str:
    """家具业务专用caption构建"""
    caption_parts = ["以下是该家具商品的已知信息："]

    key_fields = {
        "风格": item.get("风格", ""),
        "子类": item.get("子类", ""),
        "商品名": item.get("商品名", ""),
        "材质规格": item.get("subtitle", ""),
        "关键词": item.get("keyword", "")
    }

    for field, value in key_fields.items():
        if value:
            caption_parts.append(f"- {field}: {value}")

    caption_parts.extend([
        "",
        "请分析家具产品图像，重点提取：",
        "1. 材质工艺和质感特征",
        "2. 设计风格和美学元素",
        "3. 功能特点和使用场景",
        "4. 尺寸规格和空间适配性"
    ])

    return "\n".join(caption_parts)


def _build_toilet_caption(item: dict) -> str:
    """马桶业务专用caption构建"""
    caption_parts = ["以下是该马桶商品的已知信息："]

    key_fields = {
        "品牌": item.get("品牌", ""),
        "型号": item.get("型号", ""),
        "商品名": item.get("商品名", ""),
        "功能": item.get("功能", ""),
        "规格": item.get("规格", ""),
        "适用场景": item.get("适用场景", ""),
        "关键词": item.get("keyword", "")
    }

    for field, value in key_fields.items():
        if value:
            caption_parts.append(f"- {field}: {value}")

    caption_parts.extend([
        "",
        "请分析马桶产品图像，重点提取：",
        "1. 外观设计和材质（陶瓷、表面处理等）",
        "2. 功能特点（冲洗方式、节水技术、智能功能等）",
        "3. 安装规格（坑距、尺寸、重量等）",
        "4. 使用体验（舒适度、操作便利性等）"
    ])

    return "\n".join(caption_parts)


def _build_electronics_caption(item: dict) -> str:
    """电器业务专用caption构建"""
    caption_parts = ["以下是该电器商品的已知信息："]

    key_fields = {
        "品牌": item.get("品牌", ""),
        "型号": item.get("型号", ""),
        "商品名": item.get("商品名", ""),
        "功能特点": item.get("功能特点", ""),
        "技术参数": item.get("技术参数", ""),
        "适用场景": item.get("适用场景", "")
    }

    for field, value in key_fields.items():
        if value:
            caption_parts.append(f"- {field}: {value}")

    caption_parts.extend([
        "",
        "请分析电器产品图像，重点提取：",
        "1. 外观设计和做工品质",
        "2. 技术特点和创新功能",
        "3. 操作界面和使用便利性",
        "4. 适用场景和性能表现"
    ])

    return "\n".join(caption_parts)


def _build_generic_caption(item: dict) -> str:
    """通用caption构建"""
    caption_parts = ["以下是该商品的已知信息："]

    # 通用字段
    common_fields = ["商品名", "品牌", "型号", "规格", "功能", "关键词", "keyword"]

    for field in common_fields:
        value = item.get(field, "")
        if value:
            caption_parts.append(f"- {field}: {value}")

    caption_parts.extend([
        "",
        "请分析产品图像，重点提取：",
        "1. 外观设计和材质特征",
        "2. 功能特点和技术亮点",
        "3. 使用场景和适用性",
        "4. 品质和工艺表现"
    ])

    return "\n".join(caption_parts)


rag_response =  """你是侘界家具的智能对话机器人，请基于知识图谱信息，为用户提供专业的家具产品问答，并帮助用户做出最佳购买决策。
    **用户需求**：{user_query}
    特别注意：图片显示只能是后缀为.jpg/.png的格式

    **推荐要求**：
    1. **产品匹配**：根据用户需求精准匹配合适的家具产品
    2. **详细介绍**：
       - 材质工艺：木材种类、石材类型、制作工艺
       - 设计特色：风格分类、美学元素、设计亮点
       - 功能特点：使用场景、实用功能、人体工学
       - 规格参数：准确的尺寸数据、空间适配性
    
    3. **专业建议**：
       - 选购要点和注意事项
       - 保养维护指导
       - 空间搭配建议
       - 价格性价比分析
    
    4. **图片展示**：提供高质量的产品图片（确保不对原始图片链接进行修改，返回图片markdown url）
    
    5. **格式要求**：
       - 使用清晰的标题结构
       - 重要信息加粗标记
       - 分段明确，便于阅读
       - 语言专业但通俗易懂
    
    请提供详细、准确、实用的中文家具推荐方案，帮助用户做出最佳购买决策。"""