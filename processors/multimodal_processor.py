import json
import re
from typing import Dict, List, Any, Optional


class MultiModalProcessor:
    """多模态处理器 - 集成实习生的处理逻辑"""

    def __init__(self, business_id: str):
        self.business_id = business_id

    def generate_product_caption(self, item: Dict[str, Any]) -> str:
        """生成产品描述 - 借鉴实习生的逻辑"""
        # 动态提取字段映射
        field_mappings = self._get_business_field_mappings()

        fields = []
        for display_name, field_key in field_mappings.items():
            value = item.get(field_key)
            if value:
                fields.append(f"- {display_name}: {value}")

        fields_str = "\n".join(fields) if fields else "无可用字段信息"

        # 使用实习生优化过的提示词结构
        caption = f"""以下是该商品的已知信息（仅供缺失字段回填，禁止臆造）：
{fields_str}

请分析产品图像，重点提取：
1. 材质工艺和质感特征
2. 设计风格和美学元素
3. 功能特点和使用场景
4. 尺寸规格和空间适配性
"""
        return caption

    def _get_business_field_mappings(self) -> Dict[str, str]:
        """获取业务字段映射"""
        mappings = {
            "furniture": {
                "风格": "风格",
                "子类": "子类",
                "商品名": "商品名",
                "规格说明": "subtitle",
                "关键词": "keyword"
            },
            "toilet": {
                "产品名称": "产品名称",
                "品牌": "品牌",
                "规格参数": "规格参数",
                "产品特点": "产品特点"
            }
        }
        return mappings.get(self.business_id, {})

    def build_modal_content(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """构建多模态内容 - 参考实习生的结构"""
        # 处理图片路径结构
        img_path = self._build_image_structure(item)

        # 生成caption
        caption = self.generate_product_caption(item)

        # 构建modal_content
        modal_content = {
            "img_path": img_path,
            "img_caption": [caption],
            "img_footnote": []
        }

        return modal_content

    def _build_image_structure(self, item: Dict[str, Any]) -> Dict[str, List[str]]:
        """构建图片结构"""
        # 支持多种图片字段格式
        cover_pic = ""
        detail_images = []

        # 处理封面图
        if item.get("cover_pic"):
            cover_pic = item["cover_pic"]
        elif item.get("main_image"):
            cover_pic = item["main_image"]

        # 处理详情图
        if item.get("detail_images"):
            if isinstance(item["detail_images"], list):
                detail_images = item["detail_images"]
            elif isinstance(item["detail_images"], str):
                detail_images = [item["detail_images"]]

        return {
            "cover_pic": cover_pic,
            "detail_images": detail_images
        }

    def parse_response_with_fallback(self, response: str) -> Dict[str, Any]:
        """响应解析 - 使用实习生的robust解析逻辑"""
        # 尝试多种解析策略
        for strategy in [self._try_json_parse, self._try_regex_parse, self._fallback_parse]:
            try:
                result = strategy(response)
                if result:
                    return result
            except Exception as e:
                print(f"解析策略失败: {e}")
                continue

        # 最终兜底
        return {"error": "解析失败", "raw_response": response}

    def _try_json_parse(self, response: str) -> Optional[Dict[str, Any]]:
        """尝试JSON解析"""
        # 提取JSON代码块
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # 直接解析
        return json.loads(response)

    def _try_regex_parse(self, response: str) -> Optional[Dict[str, Any]]:
        """正则表达式解析"""
        # 提取关键字段
        patterns = {
            "description": r'"(?:detailed_description|description)":\s*"([^"]*(?:\\.[^"]*)*)"',
            "entity_name": r'"entity_name":\s*"([^"]*(?:\\.[^"]*)*)"',
            "summary": r'"summary":\s*"([^"]*(?:\\.[^"]*)*)"'
        }

        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, response)
            if match:
                result[key] = match.group(1)

        return result if result else None

    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """兜底解析"""
        return {
            "description": response[:500] + "..." if len(response) > 500 else response,
            "entity_info": {
                "entity_name": "解析失败的内容",
                "entity_type": "unknown",
                "summary": "内容解析失败，使用原始响应"
            }
        }