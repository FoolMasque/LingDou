# api/models.py
"""
API数据模型
"""
from datetime import datetime

from pydantic import BaseModel,Field
from typing import Optional, List,Dict,Any

class ProcessRequest(BaseModel):
    business_id: str
    json_file: str

class QueryRequest(BaseModel):
    business_id: str
    query: str
    mode: str = "hybrid"

    image_urls: Optional[List[str]] = None  # 新增：支持多张图片URL
    image_base64_list: Optional[List[str]] = None  # 新增：支持base64编码的图片

    # 历史记录
    conversation_id: Optional[str] = Field(None, description="会话ID")
    # history: Optional[List[ChatMessage]] = Field(None, description="历史消息")
    max_history: int = Field(5, description="最大历史消息数")
    streaming: bool = False  # True=流式, False=一次性返回
    
    metadata: Optional[Dict[str, Any]] = Field(None, description="新建会话时附加的初始化元数据 (例如 user_persona)")

class QueryResponse(BaseModel):
    success: bool
    query: str
    result: str
    processing_time: float
    conversation_id: Optional[str] = None
    images: Optional[list] = []
    user_images_count: Optional[int] = 0
    library_images_count: Optional[int] = 0


class ChatMessage(BaseModel):
    """对话消息"""
    role: str = Field(..., description="角色: user/assistant")
    content: str = Field(..., description="消息内容")
    images: Optional[List[str]] = Field(None, description="附带的图片")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

class ConversationCreate(BaseModel):
    """创建会话请求"""
    business_id: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversationListRequest(BaseModel):
    """列出会话请求"""
    business_id: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = Field(10, ge=1, le=100)

class ConversationExportRequest(BaseModel):
    """导出会话请求"""
    format: str = Field("json", description="导出格式: json/text")

class BusinessConfigUpdate(BaseModel):
    """更新业务配置请求"""
    response_instruction: Optional[str] = None
    field_mapping: Optional[Dict[str, str]] = None
    caption_template: Optional[str] = None
    caption_instructions: Optional[List[str]] = None
    vision_prompt_template: Optional[str] = None
    system_prompt_template: Optional[str] = None
