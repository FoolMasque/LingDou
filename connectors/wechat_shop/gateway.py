import os
import sqlite3
import json
import logging
from typing import List, Dict, Tuple, Optional
import urllib.parse
import hashlib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import httpx
import openai

# --- Configuration ---
# Absolute paths to ensure it works regardless of where it's executed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "wechat_history.db")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API URL from environment or hardcode defaults
LINGDOU_API_URL = os.getenv("LINGDOU_API_URL", "http://47.100.14.93:8008/api/query")

# Hardcode or use specific GATEWAY_ prefix to avoid global ENV conflicts
OPENAI_API_KEY = os.getenv("GATEWAY_OPENAI_API_KEY", "sk-e5bab9b0d89b42759f0832de6c2ece07")  
OPENAI_BASE_URL = os.getenv("GATEWAY_OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
INTENT_MODEL = os.getenv("GATEWAY_INTENT_MODEL", "qwen-turbo") 

app = FastAPI(title="WeChat Shop Intent Gateway")

# --- Database & Context Management ---

def init_db():
    """Initialize the SQLite database for conversation history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            openid TEXT NOT NULL,
            role TEXT NOT NULL,  -- 'user' or 'assistant'
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_openid ON history(openid)')
    
    # New table for mapping openid to LingDou conversation_id with timestamp
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS id_mapping (
            openid TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            last_active DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

async def get_or_create_lingdou_conversation(openid: str) -> str:
    """Gets valid conversation_id. Creates new one if none exists or if older than 2 hours."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Check if a recent conversation exists (within 2 hours)
    cursor.execute('''
        SELECT conversation_id 
        FROM id_mapping 
        WHERE openid = ? AND datetime(last_active) >= datetime('now', '-2 hours')
    ''', (openid,))
    row = cursor.fetchone()
    
    if row:
        # Update last_active timestamp
        cursor.execute('UPDATE id_mapping SET last_active = datetime("now") WHERE openid = ?', (openid,))
        conn.commit()
        conn.close()
        return row[0]
        
    conn.close()
    
    # If not found or expired, create a new one via LingDou API
    async with httpx.AsyncClient() as client:
        try:
            # We assume LingDou runs on localhost:8008 based on your config
            create_url = f"http://47.100.14.93:8008/api/conversations/new?business_id=wechat_shop&user_id={openid}"
            resp = await client.post(create_url)
            resp.raise_for_status()
            
            data = resp.json()
            new_conv_id = data.get("conversation_id")
            
            if new_conv_id:
                # Save mapping - Use REPLACE in case the openid exists but was expired
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO id_mapping (openid, conversation_id, last_active) 
                    VALUES (?, ?, datetime('now'))
                ''', (openid, new_conv_id))
                conn.commit()
                conn.close()
                return new_conv_id
            else:
                logger.error("Failed to get 'conversation_id' from new conversation response")
                return ""
                
        except Exception as e:
            logger.error(f"Error creating LingDou conversation: {e}")
            return ""

def get_recent_history(openid: str, limit: int = 6) -> str:
    """Fetch the recent N conversation turns for a user, formatted as text."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Order by ID descending to get latest, then reverse in Python to maintain chronological order
    cursor.execute('''
        SELECT role, content FROM history 
        WHERE openid = ? 
        ORDER BY id DESC LIMIT ?
    ''', (openid, limit))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "无历史记录"
    
    # Reverse to chronological order (oldest to newest)
    rows.reverse()
    
    history_lines = []
    for role, content in rows:
        role_name = "用户" if role == "user" else "客服"
        history_lines.append(f"{role_name}: {content}")
        
    return "\n".join(history_lines)

def add_message(openid: str, role: str, content: str):
    """Add a single message to the history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (openid, role, content) 
        VALUES (?, ?, ?)
    ''', (openid, role, content))
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# --- Intent Recognition ---

class IntentResult(BaseModel):
    intent: str
    reply: str = ""

async def recognize_intent_and_extract(query: str, history_text: str) -> IntentResult:
    """
    Call the minimal LLM to determine the user's intent.
    Returns an intent string: "product", "order", or "chat".
    """
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    
    system_prompt = """你是一个智能客服意图识别引擎-灵豆。
你的任务是根据【历史对话】和【最新用户输入】，判断用户的核心意图。

意图只允许是以下三种之一：
1. "product": 商品咨询（问价格、参数、口感、库存、推荐、怎么买等与商品销售相关的问题）。
2. "order": 订单/物流查询（问发货没、到哪了、单号多少、退换货状态等售后/物流问题）。
3. "chat": 闲聊/其他（闲聊、打招呼、无意义输入，或者完全无关的问题）。

如果意图是 "chat"，请顺便生成一句简短的拟人化【回复(reply)】引导用户询问商品或订单相关的问题。
如果意图是 "product" 或 "order"，回复必须**完全留空**（即 reply: ""）。绝对不要编造不存在的订单号或发货状态！！！

输出必须是一个合法的 JSON 对象，不包含 Markdown 标记，格式如下：
{"intent": "product|order|chat", "reply": "你的回复"}
"""
    
    user_prompt = f"【历史对话】\n{history_text}\n\n【最新用户输入】\n{query}"
    
    try:
        response = await client.chat.completions.create(
            model=INTENT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        return IntentResult(intent=data.get("intent", "product"), reply=data.get("reply", ""))
        
    except Exception as e:
        logger.error(f"Intent recognition failed: {e}")
        # Default to product inquiry to utilize LingDou's RAG as fallback
        return IntentResult(intent="product")

# --- API Models ---

class WeChatWebhookRequest(BaseModel):
    openid: str
    query: str
    tracking_number: Optional[str] = None
    com: Optional[str] = "yuantong"  # Carrier code, default to yuantong
    # In a real environment, you'd have msg_id, event_type, etc.

class GatewayResponse(BaseModel):
    reply: str
    intent_detected: str
    images: Optional[List[str]] = []

# --- Logistics API Handlers ---

async def fetch_logistics_info(tracking_number: str, com: str = "yuantong") -> dict:
    """Fetch real logistics data from kuaidi100 API."""
    key = 'EbZqPMOG1512'  # 客户授权key
    customer = '7E82DA37D3BF142CABAA0F1FEEBB0374'  # 查询公司编号
    url = 'https://poll.kuaidi100.com/poll/query.do'
    
    param = {
        'com': com,
        'num': tracking_number,
        'phone': '',
        'from': '',
        'to': '',
        'resultv2': '1',
        'show': '0',
        'order': 'desc'
    }
    
    param_str = json.dumps(param)
    # 签名加密
    temp_sign = param_str + key + customer
    md = hashlib.md5()
    md.update(temp_sign.encode())
    sign = md.hexdigest().upper()
    
    request_data = {
        'customer': customer,
        'param': param_str,
        'sign': sign
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # Send as Form Data
            resp = await client.post(url, data=request_data, timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch logistics for {tracking_number}: {e}")
            return {"message": "获取物流信息失败"}

async def generate_order_reply(query: str, tracking_number: str, logistics_data: dict) -> str:
    """Use LLM to generate a natural response based on raw API data."""
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    system_prompt = """你是一个智能客服-灵豆。
用户正在查询订单/物流信息。
以下是快递 API 返回的真实物流 JSON 数据。
请你根据用户的诉求，将 JSON 里的物流轨迹提炼成一段自然、简短、有温度的客服回复。
切记：只能依据提供的 JSON 数据回答，绝不能自己编造任何状态或时间！如果查不到，就温柔地告知用户暂时查不到。"""

    user_prompt = f"【用户问题】\n{query}\n\n【物流单号】\n{tracking_number}\n\n【物流数据】\n{json.dumps(logistics_data, ensure_ascii=False)}"
    
    try:
        response = await client.chat.completions.create(
            model=INTENT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Failed to generate order reply: {e}")
        return f"亲，您的物流单号 {tracking_number} 物流信息我暂时查不到哦，请稍后再试~"

# --- Main Route ---

@app.post("/webhook", response_model=GatewayResponse)
async def wechat_shop_webhook(request: WeChatWebhookRequest):
    """
    The main entry point for WeChat Shop messages.
    """
    openid = request.openid
    query = request.query
    
    logger.info(f"[{openid}] Received query: {query}")
    
    # 1. Store user message in local history
    add_message(openid, "user", query)
    
    # 2. Retrieve recent history for context
    history_text = get_recent_history(openid, limit=6)
    logger.info(f"[{openid}] Context: {history_text}")
    
    # 3. Intent Recognition
    intent_res = await recognize_intent_and_extract(query, history_text)
    logger.info(f"[{openid}] Intent Detected: {intent_res.intent}")
    
    final_reply = ""
    extracted_images = []
    
    # 4. Routing Logic
    if intent_res.intent == "order":
        # 拦截：订单咨询逻辑
        logger.info(f"[{openid}] Routing to ORDER handler.")
        if request.tracking_number:
            logger.info(f"[{openid}] Querying logistics for tracking_number: {request.tracking_number} (carrier: {request.com})")
            logistics_data = await fetch_logistics_info(request.tracking_number, request.com)
            final_reply = await generate_order_reply(query, request.tracking_number, logistics_data)
        else:
            final_reply = "亲，请问您可以提供一下您的订单编号或物流单号吗？灵豆马上帮您查询哟~"
        
    elif intent_res.intent == "chat":
        # 拦截：闲聊兜底逻辑
        logger.info(f"[{openid}] Routing to CHAT handler.")
        final_reply = intent_res.reply or "亲，我是专属客服灵豆，您可以直接问我商品相关的问题哦~ 😊"
        
    else:
        # Default/Expected: 商品咨询 -> 透传给 LingDou
        logger.info(f"[{openid}] Routing to LINGDOU RAG Engine.")
        
        # Act as an HTTP client calling LingDou
        async with httpx.AsyncClient() as client:
            try:
                # Retrieve the real LingDou conversation_id instead of using openid
                lingdou_conv_id = await get_or_create_lingdou_conversation(openid)
                
                payload = {
                    "query": query,
                    "business_id": "wechat_shop",
                    "conversation_id": lingdou_conv_id,  
                    "streaming": False
                }
                
                resp = await client.post(LINGDOU_API_URL, json=payload, timeout=60.0)
                resp.raise_for_status()
                
                lingdou_data = resp.json()
                final_reply = lingdou_data.get("result", "抱歉，系统开小差了，请稍后再试~")
                extracted_images = lingdou_data.get("images", [])
                
            except httpx.RequestError as e:
                logger.error(f"Failed to communicate with LingDou API: {e}")
                final_reply = "抱歉，灵豆的知识库网络有点波动，请稍后再试~"
            except httpx.HTTPStatusError as e:
                 logger.error(f"LingDou API returned error: {e.response.text}")
                 final_reply = "抱歉，知识库暂不可用~"

    # 5. Store assistant reply in local history
    add_message(openid, "assistant", final_reply)
    
    # 6. Return standard response (Assume WeChat server consumes this)
    return GatewayResponse(
        reply=final_reply, 
        intent_detected=intent_res.intent,
        images=extracted_images
    )

if __name__ == "__main__":
    import uvicorn
    # Run Gateway on a different port than LingDou (8008)
    print("🚀 Starting WeChat Shop Intent Gateway on port 8009...")
    uvicorn.run(app, host="0.0.0.0", port=8009)
