# api/models.py
"""
API数据模型
"""
from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    business_id: str
    json_file: str

class QueryRequest(BaseModel):
    business_id: str
    query: str
    mode: str = "hybrid"

class QueryResponse(BaseModel):
    success: bool
    query: str
    result: str
    processing_time: float
