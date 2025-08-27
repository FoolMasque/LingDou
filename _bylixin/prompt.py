"""
Prompt templates for multimodal content processing

Contains all prompt templates used in modal processors for analyzing
different types of content (images, tables, equations, etc.)
"""

from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# System prompts for different analysis types
PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
    # "You are an expert image analyst. Provide detailed, accurate descriptions."
    "您是一位家具图像分析专家。提供准确的描述。"
)
PROMPTS["DETAIL_OCR_SYSTEM"] = (
    "您是一位专业的图像文字提取（OCR）专家。任务是从提供的图片中精准识别并提取所有可见文字，保持原有的顺序与结构。不要添加额外描述或解释，只输出识别到的文字内容。"
)


PROMPTS["IMAGE_ANALYSIS_FALLBACK_SYSTEM"] = (
    "You are an expert image analyst. Provide detailed analysis based on available information."
)
PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
    "You are an expert data analyst. Provide detailed table analysis with specific insights."
)
PROMPTS["EQUATION_ANALYSIS_SYSTEM"] = (
    "You are an expert mathematician. Provide detailed mathematical analysis."
)
PROMPTS["GENERIC_ANALYSIS_SYSTEM"] = (
    "You are an expert content analyst specializing in {content_type} content."
)

#Image analysis prompt template
# PROMPTS[
#     "vision_prompt"
# ] = """Please analyze this image in detail and provide a JSON response with the following structure:

# {{
#     "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
#     - Describe the overall composition and layout
#     - Identify all objects, people, text, and visual elements
#     - Explain relationships between elements
#     - Note colors, lighting, and visual style
#     - Describe any actions or activities shown
#     - Include technical details if relevant (charts, diagrams, etc.)
#     - Always use specific names instead of pronouns",
#     "entity_info": {{
#         "entity_name": "{entity_name}",
#         "entity_type": "image",
#         "summary": "concise summary of the image content and its significance (max 100 words)"
#     }}
# }}

# Additional context:
# - Image Path: {image_path}
# - Captions: {captions}
# - Footnotes: {footnotes}

# Focus on providing accurate, detailed visual analysis that would be useful for knowledge retrieval."""

# #Image analysis prompt template  ####################可行
# PROMPTS[
#     "vision_prompt"
# ] = """你是“商品信息抽取器”。忽略背景与上下文，只关注商品本体。提供具有以下结构的JSON响应:

# {{
#     "detailed_description": "若图中有文字，请一字不落提取文字，并用一句话描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词；",
#     "entity_info": {{
#         "entity_name": "{entity_name}",
#         "entity_type": "image",
#         "summary": "图像内容、其重要性及与周围内容关系的简明摘要（最多100字）"
#     }}
# }}

# Additional context:
# - Image Path: {image_path}
# - Captions: {captions}
# - Footnotes: {footnotes}

# 重点提供准确、详细的视觉分析，结合上下文以支持知识检索。"""

# #Image analysis prompt template  ####################可行
# PROMPTS[
#     "vision_prompt"
# ] = """你是“商品信息抽取器”。忽略背景与上下文，只关注商品本体。提供具有以下结构的JSON响应:

# {{
#     "detailed_description": "若图中有文字，请一字不落提取文字，并用一句话描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词；",
#     "entity_info": {{
#         "entity_name": "{entity_name}",
#         "entity_type": "image",
#         "summary": "图像内容、其重要性及与周围内容关系的简明摘要（最多100字）"
#     }}
# }}

# Additional context:
# - Image Path: {image_path}
# - Captions: {captions}
# - Footnotes: {footnotes}

# 重点提供准确、详细的视觉分析，结合上下文以支持知识检索。"""


#Image analysis prompt template  ####################尝试
PROMPTS[
    "vision_prompt"
] = """你是“OCR + 商品信息抽取器”。只输出一个 JSON 对象：

{{
    "ocr_text": "逐字提取图像中出现的所有文字，按行保留换行和标点；不得改写、总结或省略；如果没有文字，填空字符串",
    "description": "若图中有文字，请一字不落提取文字，并用一句话描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词；",
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

重点提供准确的视觉分析，结合上下文以支持知识检索。"""





# PROMPTS["vision_prompt"] = """你是“OCR + 商品信息抽取器”。只输出一个 JSON 对象：

# {{
#   "ocr_text": "逐字提取图像中出现的所有文字，按行保留换行和标点；不得改写、总结或省略；如果没有文字，填空字符串",
#   "entity_info": {{
#     "entity_name": "{entity_name}",
#     "entity_type": "image",
#     "summary": "用不超过100字概括商品（而非环境）的关键信息"
#   }},
#   "description": "≤50字的单句，只描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词"
# }}

# 注意：
# - 先完成 ocr_text（全文逐字），再生成 description（单句商品概述）。
# - 禁词仅作用于 description，**不作用于 ocr_text**。
# - 只能输出 JSON；JSON 之外的任何字符视为错误。

# 上下文（可选，仅用于理解商品，不可写入 ocr_text）:
# {context}

# 图像信息：
# - Image Path: {image_path}
# - Captions: {captions}
# - Footnotes: {footnotes}
# """





# #Image analysis prompt template
# PROMPTS[
#     "vision_prompt"
# ] = """你是“商品信息抽取器”。忽略背景与上下文，只关注商品本体。提供具有以下结构的JSON响应:

# {{
#     "style": "提取的风格（图片没出现则用提供信息）",
#     "category": "提取的子类（图片没出现则用提供信息）",
#     "name": "提取的商品名（图片没出现则用提供信息）",
#     "subtitle": "提取的副标题（图片没出现则用提供信息）",
#     "keyword": "提取的关键词（图片没出现则用提供信息）",
#     "description": "≤50字，仅描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词"
#     "entity_info": {{
#         "entity_name": "{entity_name}",
#         "entity_type": "image",
#         "summary": "图像内容、其重要性及与周围内容关系的简明摘要（最多100字）"
#     }}
# }}

# Additional context:
# - Image Path: {image_path}
# - Captions: {captions}
# - Footnotes: {footnotes}

# 重点提供准确的视觉分析，结合上下文以支持知识检索。"""



# # Image analysis prompt with context support
# PROMPTS[
#     "vision_prompt_with_context"
# ] = """Please analyze this image in detail, considering the surrounding context. Provide a JSON response with the following structure:

# {{
#     "detailed_description": "A comprehensive and detailed visual description of the image following these guidelines:
#     - Describe the overall composition and layout
#     - Identify all objects, people, text, and visual elements
#     - Explain relationships between elements and how they relate to the surrounding context
#     - Note colors, lighting, and visual style
#     - Describe any actions or activities shown
#     - Include technical details if relevant (charts, diagrams, etc.)
#     - Reference connections to the surrounding content when relevant
#     - Always use specific names instead of pronouns",
#     "entity_info": {{
#         "entity_name": "{entity_name}",
#         "entity_type": "image",
#         "summary": "concise summary of the image content, its significance, and relationship to surrounding content (max 100 words)"
#     }}
# }}

# Context from surrounding content:
# {context}

# Image details:
# - Image Path: {image_path}
# - Captions: {captions}
# - Footnotes: {footnotes}

# Focus on providing accurate, detailed visual analysis that incorporates the context and would be useful for knowledge retrieval."""

# # Image analysis prompt with context support
# PROMPTS[
#     "vision_prompt_with_context"
# ] = """请详细分析此图像，并考虑周围上下文，提供以下结构的 JSON 响应：
# {{
#     "detailed_description": "按照以下指导方针对图像进行全面且详细的视觉描述：
#     - 描述整体构图和布局
#     - 识别所有物体、人物、文本和视觉元素
#     - 解释元素之间的关系以及它们与周围上下文的关联
#     - 记录颜色、照明和视觉风格
#     - 描述显示的任何动作或活动
#     - 如相关，包含技术细节（如图表、示意图等）
#     - 如相关，引用与周围内容的联系
#     - 始终使用具体名称而非代词",
#     "entity_info": {{
#         "entity_name": "{entity_name}",
#         "entity_type": "image",
#         "summary": "图像内容、其重要性及与周围内容关系的简明摘要（最多100字）"
#     }}
# }}

# Context from surrounding content:
# {context}

# Image details:
# - Image Path: {image_path}
# - Captions: {captions}
# - Footnotes: {footnotes}

# 重点提供准确、详细的视觉分析，结合上下文以支持知识检索。"""

# # Image analysis prompt with context support   ##############################可行
# PROMPTS[
#     "vision_prompt_with_context"
# ] = """你是“商品信息抽取器”。忽略背景与上下文，只关注商品本体。提供具有以下结构的JSON响应:
# {{
#     "detailed_description": "若图中有文字，请一字不落提取文字，并用一句话描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词；",
#     "entity_info": {{
#         "entity_name": "{entity_name}",
#         "entity_type": "image",
#         "summary": "图像内容、其重要性及与周围内容关系的简明摘要（最多100字）"
#     }}
# }}

# Context from surrounding content:
# {context}

# Image details:
# - Image Path: {image_path}
# - Captions: {captions}
# - Footnotes: {footnotes}

# 重点提供准确、详细的视觉分析，结合上下文以支持知识检索。"""



# Image analysis prompt with context support   ##############################尝试
PROMPTS[
    "vision_prompt_with_context"
] = """你是“OCR + 商品信息抽取器”。只输出一个 JSON 对象：
{{
    "ocr_text": "逐字提取图像中出现的所有文字，按行保留换行和标点；不得改写、总结或省略；如果没有文字，填空字符串",
    "description": "若图中有文字，请一字不落提取文字，并用一句话描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词；",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "图像内容、其重要性及与周围内容关系的简明摘要（最多100字）"
    }}
}}

Context from surrounding content:
{context}

Image details:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

重点提供准确、详细的视觉分析，结合上下文以支持知识检索。"""




# Image analysis prompt with cover   ##############################尝试
PROMPTS[
    "vision_prompt_cover"
] = """商品信息抽取器”。只输出一个 JSON 对象：
{{
    "text": "",
    "description": "用一句话描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词；",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "图像内容、其重要性及与周围内容关系的简明摘要（最多100字）"
    }}
}}

Context from surrounding content:
{context}

Image details:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

重点提供准确的视觉分析，结合上下文以支持知识检索。"""


# Image analysis prompt with detail   ##############################尝试
PROMPTS[
    "detail_prompt"
] = """你是“OCR + 商品信息抽取器”。只输出一个 JSON 对象：
{{
    "text": "逐字提取图像中出现的所有文字，按行保留换行和标点；不得改写、总结或省略；如果没有文字，填空字符串",
    "description": "",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "图像内容、其重要性及与周围内容关系的简明摘要（最多100字）"
    }}
}}

Context from surrounding content:
{context}

Image details:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

重点提供准确的视觉分析，结合上下文以支持知识检索。"""

# PROMPTS["vision_prompt_with_context"] = """你是“OCR + 商品信息抽取器”。只输出一个 JSON 对象：

# {
#   "ocr_text": "逐字提取图像中出现的所有文字，按行保留换行和标点；不得改写、总结或省略；如果没有文字，填空字符串",
#   "entity_info": {
#     "entity_name": "{entity_name}",
#     "entity_type": "image",
#     "summary": "用不超过100字概括商品（而非环境）的关键信息"
#   },
#   "description": "≤50字的单句，只描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词"
# }

# 注意：
# - 先完成 ocr_text（全文逐字），再生成 description（单句商品概述）。
# - 禁词仅作用于 description，**不作用于 ocr_text**。
# - 只能输出 JSON；JSON 之外的任何字符视为错误。

# 上下文（可选，仅用于理解商品，不可写入 ocr_text）:
# {context}

# 图像信息：
# - Image Path: {image_path}
# - Captions: {captions}
# - Footnotes: {footnotes}
# """






# # Image analysis prompt with context support
# PROMPTS[
#     "vision_prompt_with_context"
# ] = """你是“商品信息抽取器”。忽略背景与上下文，只关注商品本体。
# {{
#     "style": "提取的风格（图片没出现则用提供信息）",
#     "category": "提取的子类（图片没出现则用提供信息）",
#     "name": "提取的商品名（图片没出现则用提供信息）",
#     "subtitle": "提取的副标题（图片没出现则用提供信息）",
#     "keyword": "提取的关键词（图片没出现则用提供信息）",
#     "description": "≤50字，仅描述商品本体的材质/颜色/形态/功能；禁止出现：背景/墙/地面/窗帘/植物/环境/灯/灯光/射灯/空间/构图/场景/瓷砖/壁灯 等词"
#     "entity_info": {{
#         "entity_name": "{entity_name}",
#         "entity_type": "image",
#         "summary": "图像内容、其重要性及与周围内容关系的简明摘要（最多100字）"
#     }}
# }}

# Context from surrounding content:
# {context}

# Image details:
# - Image Path: {image_path}
# - Captions: {captions}
# - Footnotes: {footnotes}

# 重点提供准确的视觉分析，结合上下文以支持知识检索。"""

# # Image analysis prompt with text fallback
# PROMPTS["text_prompt"] = """Based on the following image information, provide analysis:

# Image Path: {image_path}
# Captions: {captions}
# Footnotes: {footnotes}

# {vision_prompt}"""


# Image analysis prompt with text fallback
PROMPTS["text_prompt"] = """根据以下图像信息提供分析：

Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

{vision_prompt}"""


# Table analysis prompt template
PROMPTS[
    "table_prompt"
] = """Please analyze this table content and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented
    Always use specific names and values instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose and key findings (max 100 words)"
    }}
}}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Focus on extracting meaningful insights and relationships from the tabular data."""

# Table analysis prompt with context support
PROMPTS[
    "table_prompt_with_context"
] = """Please analyze this table content considering the surrounding context, and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization
    - Column headers and their meanings
    - Key data points and patterns
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented in relation to surrounding context
    - How the table supports or illustrates concepts from the surrounding content
    Always use specific names and values instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose, key findings, and relationship to surrounding content (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Table Information:
Image Path: {table_img_path}
Caption: {table_caption}
Body: {table_body}
Footnotes: {table_footnote}

Focus on extracting meaningful insights and relationships from the tabular data in the context of the surrounding content."""

# Equation analysis prompt template
PROMPTS[
    "equation_prompt"
] = """Please analyze this mathematical equation and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions
    - Mathematical operations and functions used
    - Application domain and context
    - Physical or theoretical significance
    - Relationship to other mathematical concepts
    - Practical applications or use cases
    Always use specific mathematical terminology.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose and significance (max 100 words)"
    }}
}}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

Focus on providing mathematical insights and explaining the equation's significance."""

# Equation analysis prompt with context support
PROMPTS[
    "equation_prompt_with_context"
] = """Please analyze this mathematical equation considering the surrounding context, and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the equation including:
    - Mathematical meaning and interpretation
    - Variables and their definitions in the context of surrounding content
    - Mathematical operations and functions used
    - Application domain and context based on surrounding material
    - Physical or theoretical significance
    - Relationship to other mathematical concepts mentioned in the context
    - Practical applications or use cases
    - How the equation relates to the broader discussion or framework
    Always use specific mathematical terminology.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "equation",
        "summary": "concise summary of the equation's purpose, significance, and role in the surrounding context (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Equation Information:
Equation: {equation_text}
Format: {equation_format}

Focus on providing mathematical insights and explaining the equation's significance within the broader context."""

# Generic content analysis prompt template
PROMPTS[
    "generic_prompt"
] = """Please analyze this {content_type} content and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose and key points (max 100 words)"
    }}
}}

Content: {content}

Focus on extracting meaningful information that would be useful for knowledge retrieval."""

# Generic content analysis prompt with context support
PROMPTS[
    "generic_prompt_with_context"
] = """Please analyze this {content_type} content considering the surrounding context, and provide a JSON response with the following structure:

{{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance in relation to surrounding content
    - How this content connects to or supports the broader discussion
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose, key points, and relationship to surrounding context (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Content: {content}

Focus on extracting meaningful information that would be useful for knowledge retrieval and understanding the content's role in the broader context."""

# Modal chunk templates
PROMPTS["image_chunk"] = """
Image Content Analysis:
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Visual Analysis: {enhanced_caption}"""

PROMPTS["table_chunk"] = """Table Analysis:
Image Path: {table_img_path}
Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}"""

PROMPTS["equation_chunk"] = """Mathematical Equation Analysis:
Equation: {equation_text}
Format: {equation_format}

Mathematical Analysis: {enhanced_caption}"""

PROMPTS["generic_chunk"] = """{content_type} Content Analysis:
Content: {content}

Analysis: {enhanced_caption}"""

# Query-related prompts
PROMPTS["QUERY_IMAGE_DESCRIPTION"] = (
    "Please briefly describe the main content, key elements, and important information in this image."
)

PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"] = (
    "You are a professional image analyst who can accurately describe image content."
)

PROMPTS[
    "QUERY_TABLE_ANALYSIS"
] = """Please analyze the main content, structure, and key information of the following table data:

Table data:
{table_data}

Table caption: {table_caption}

Please briefly summarize the main content, data characteristics, and important findings of the table."""

PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"] = (
    "You are a professional data analyst who can accurately analyze table data."
)

PROMPTS[
    "QUERY_EQUATION_ANALYSIS"
] = """Please explain the meaning and purpose of the following mathematical formula:

LaTeX formula: {latex}
Formula caption: {equation_caption}

Please briefly explain the mathematical meaning, application scenarios, and importance of this formula."""

PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"] = (
    "You are a mathematics expert who can clearly explain mathematical formulas."
)

PROMPTS[
    "QUERY_GENERIC_ANALYSIS"
] = """Please analyze the following {content_type} type content and extract its main information and key features:

Content: {content_str}

Please briefly summarize the main characteristics and important information of this content."""

PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"] = (
    "You are a professional content analyst who can accurately analyze {content_type} type content."
)

PROMPTS["QUERY_ENHANCEMENT_SUFFIX"] = (
    "\n\nPlease provide a comprehensive answer based on the user query and the provided multimodal content information."
)
