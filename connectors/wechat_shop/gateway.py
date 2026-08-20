import os
import sqlite3
import json
import logging
from typing import List, Dict, Tuple, Optional
import urllib.parse
import hashlib
import traceback
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
LINGDOU_API_URL = os.getenv("LINGDOU_API_URL", "http://47.100.14.93:8008/api/query") # localhost 47.100.14.93

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
# --- Configuration Cache / Shop Config Fetching ---
# 既然不能硬挂载，且未来会无限扩展，最优雅的解法是：
# 网关（Gateway）保持"无状态化"，由外部请求的 URL 路径来提供所有动态关系。
# (此段原硬编码字典已删除)
# ---

async def get_or_create_lingdou_conversation(openid: str, business_id: str) -> str:
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
            # localhost
            create_url = f"http://47.100.14.93:8008/api/conversations/new?business_id={business_id}&user_id={openid}"
            resp = await client.post(create_url)
            resp.raise_for_status()
            
            data = resp.json()
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
    sub_intent: Optional[str] = None
    reply: str = ""
    order_id: Optional[str] = None
    complaint_summary: Optional[str] = None

async def recognize_intent_and_extract(query: str, history_text: str) -> IntentResult:
    """
    Call the minimal LLM to determine the user's intent.
    Returns an intent string: "product", "order", or "chat".
    """
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    
    system_prompt = """你是一个专业的电商智能客服意图识别引擎-灵豆。
你的任务是根据【历史对话】和【最新用户输入】，判断用户的核心意图。

【主意图分类】(intent只允许是以下四种之一)：
1. "product": 商品咨询（问价格、参数、口感、库存、推荐、怎么买等）。
2. "logistics": 物流客观查询（仅仅是客观询问：发货没？到哪了？单号多少？或者无前后文时客观发送单号查件）。绝对不包含带有情绪的催单！
3. "aftersale": 售后/转人工/客诉（退换货、发错货、投诉、催促发货/抱怨物流慢等）。注意：如果【历史对话】中客服正在请求用户提供退换货的订单凭据，此时用户单纯发送单号，主意图必须是 "aftersale"！
4. "chat": 闲聊/其他。

【售后子意图分类】(仅当intent为"aftersale"时，需进一步判断 sub_intent)：
- "damage": 破损/少件/瑕疵（商品或内外包装破损、缺件、质量问题等）。
- "wrong": 发错货（颜色/款式/型号不符）。
- "refund": 退换货/七天无理由/仅退款（拍错、多拍、不喜欢、降价等）。
- "urge": 催单/物流投诉（抱怨下单很久没发货、催发货、物流长时间停滞）。
- "other": 其他售后诉求或单纯要求转人工。如果只是补充发送订单号，请维持原来的故障子分类！

【售后回复SOP严格规则】(仅当 intent 为 "aftersale" 时执行，其他意图的 reply 必须留空 "")：
1. 必须根据用户的原话进行情绪安抚（如“真的非常抱歉让您收到破损的商品”）。**严禁使用千篇一律的废话。**
2. 提取订单号：尝试提取订单号填入 `order_id`，若无则填 null。但是**不要**在回复中索要订单号（系统会自动追加）。
3. **精准索要凭证 (核心SOP)**：
   - 如果 sub_intent 是 "damage": 必须温柔提示用户后续请准备好送拍【内外包装破损照片】和【商品破损照片】。
   - 如果 sub_intent 是 "wrong": 必须提示用户后续请准备好【收到的实物照片】。
   - 如果 sub_intent 是 "refund": 可以善意提醒用户“大部分退换货需保证商品原包装完整，不影响二次销售哦”。
   - 如果 sub_intent 是 "urge": 安抚“我们一定会催促快递站点加急处理”。
4. 绝对不允许在回复中包含任何微信号、手机号等站外联系方式！
5. 你的 reply 生成的句子须完整、温柔。对于转人工等结尾系统会自动处理。
# 6. 提取核心诉求：无论经历多少轮对话，请高度凝练归纳用户的核心诉求填入 `complaint_summary`（限20字内）。如果不是售后意图，可填 null。

输出必须是一个合法的 JSON 对象，不包含 Markdown 标记，格式如下：
# {"intent": "product|logistics|aftersale|chat", "sub_intent": "damage|wrong|refund|urge|other|null", "reply": "你的定制化回复", "order_id": "提取的订单或者null", "complaint_summary": "一句话归纳诉求"}
{"intent": "product|logistics|aftersale|chat", "sub_intent": "damage|wrong|refund|urge|other|null", "reply": "你的定制化回复", "order_id": "提取的订单或者null"}
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
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        return IntentResult(
            intent=data.get("intent", "product"), 
            sub_intent=data.get("sub_intent"),
            reply=data.get("reply", ""),
            order_id=data.get("order_id"),
            complaint_summary=data.get("complaint_summary")
        )
        
    except Exception as e:
        logger.error(f"Intent recognition failed: {e}")
        # Default to product inquiry to utilize LingDou's RAG as fallback
        return IntentResult(intent="product")

# --- API Models ---

class WeChatWebhookRequest(BaseModel):
    openid: str
    query: str
    order_id: Optional[str] = None
    com: Optional[str] = "yuantong"  # Default carrier fallback
    # In a real environment, you'd have msg_id, event_type, etc.

class GatewayResponse(BaseModel):
    reply: str
    intent_detected: str
    images: Optional[List[str]] = []
    issue_tag: Optional[str] = None

# --- WeChat API & Logistics Handlers ---

async def fetch_wechat_order_tracking(app_id: str, order_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetches WeChat access token and then fetches order details to extract waybill_id and carrier.
    Returns (waybill_id, delivery_name_or_code)
    """
    async with httpx.AsyncClient() as client:
        try:
            # 1. Get Token (Dynamic Store App ID)
            secret = os.getenv("WECHAT_STORE_SECRET", "1024%40Yinyu")
            token_url = f"http://47.100.14.93/backend//lingdou/wx_store/{app_id}/token?secret={secret}"
            token_resp = await client.get(token_url, timeout=10.0)
            token_resp.raise_for_status()
            token_data = token_resp.json()
            # Depending on actual API structure, usually it's in a 'token' or 'access_token' field
            # The backend API apparently returns a raw string or a stringified JSON
            if isinstance(token_data, str):
                try:
                    # Try to parse string as JSON just in case
                    token_json = json.loads(token_data)
                    access_token = token_json.get("access_token")
                    if not access_token:
                        data_val = token_json.get("data")
                        access_token = data_val.get("access_token") if isinstance(data_val, dict) else (data_val if isinstance(data_val, str) else None)
                except json.JSONDecodeError:
                    # If it's not JSON, assume the string itself is the raw token
                    access_token = token_data.strip()
            else:
                access_token = token_data.get("access_token")
                if not access_token:
                    data_val = token_data.get("data")
                    access_token = data_val.get("access_token") if isinstance(data_val, dict) else (data_val if isinstance(data_val, str) else None)
            
            if not access_token:
                logger.error(f"Failed to extract access_token from response: {token_data}")
                return None, None

            # 2. Get Order Details
            order_url = f"https://api.weixin.qq.com/channels/ec/order/get?access_token={access_token}"
            order_payload = {"order_id": order_id}
            order_resp = await client.post(order_url, json=order_payload, timeout=10.0)
            order_resp.raise_for_status()
            order_data = order_resp.json()
            
            error_code = order_data.get("errcode", 0)
            if error_code != 0:
                logger.error(f"WeChat API returned error {error_code}: {order_data.get('errmsg')}")
                return None, None
                
            order_info = order_data.get("order", {})
            order_detail = order_info.get("order_detail", {})
            delivery_info = order_detail.get("delivery_info", {})
            delivery_product_info = delivery_info.get("delivery_product_info", [])
            
            if delivery_product_info and len(delivery_product_info) > 0:
                first_package = delivery_product_info[0]
                waybill_id = first_package.get("waybill_id")
                # Attempt to get delivery name, but fallback to caller's provided com or yuantong
                delivery_name = first_package.get("delivery_name") or first_package.get("delivery_id")
                return waybill_id, delivery_name
            else:
                logger.info(f"Order {order_id} has no delivery info yet.")
                return None, None
                
        except Exception as e:
            logger.error(f"Error fetching WeChat order tracking: {e}\n{traceback.format_exc()}")
            return None, None

async def fetch_logistics_info(tracking_number: str, com: str = "yuantong") -> dict:
    """Fetch real logistics data from kuaidi100 API."""
    key = os.getenv("KUAIDI100_KEY", "EbZqPMOG1512")  # 客户授权 key
    customer = os.getenv("KUAIDI100_CUSTOMER", "7E82DA37D3BF142CABAA0F1FEEBB0374")  # 查询公司编号
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

@app.post("/webhook/{business_id}/{app_id}", response_model=GatewayResponse)
async def wechat_shop_webhook(business_id: str, app_id: str, request: WeChatWebhookRequest):
    """
    The main dynamic entry point for multi-tenant WeChat Shop messages.
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
    final_issue_tag = None
    
    # 4. Routing Logic
    if intent_res.intent == "logistics":
        # 拦截：物流/发货咨询逻辑
        logger.info(f"[{openid}] Routing to LOGISTICS handler.")
        if request.order_id:
            logger.info(f"[{openid}] Process order_id: {request.order_id}")
            # Step 1: Resolve order_id to waybill_id via WeChat API
            waybill_id, delivery_name = await fetch_wechat_order_tracking(app_id, request.order_id)
            
            if waybill_id:
                # Carrier Mapping Dictionary based on KuaiDi100's company codes
                carrier_map = {
                    "圆通速递": "yuantong",
                    "中通快递": "zhongtong",
                    "京东物流": "jd",
                    "韵达快递": "yunda",
                    "申通快递": "shentong",
                    "极兔速递": "jtexpress",
                    "邮政电商标快": "youzhengdsbk",
                    "德邦快递": "debangkuaidi",
                    "菜鸟速递": "danniao",
                    "邮政标准快递": "youzhengbk",
                    "中通快运": "zhongtongkuaiyun",
                    "跨越速运": "kuayue",
                    "德邦物流": "debangwuliu",
                    "京东快运": "jingdongkuaiyun",
                    "顺丰快运": "shunfengkuaiyun",
                    "安能快运": "annengwuliu",
                    "顺丰速运": "shunfeng" # Common fallback just in case
                }
                
                # Match delivery name exactly, or fallback to the provided com, then to yuantong
                carrier = carrier_map.get(delivery_name, request.com)
                carrier = carrier or "yuantong"
                
                logger.info(f"[{openid}] Querying KuaiDi100 for tracking_number: {waybill_id} (carrier: {carrier})")
                logistics_data = await fetch_logistics_info(waybill_id, carrier)
                final_reply = await generate_order_reply(query, waybill_id, logistics_data)
            else:
                final_reply = f"亲，灵豆帮您查了订单（{request.order_id}），可是暂时还没有看到物流单号呢，可能是还没发货，请稍等一下哦~"
        else:
            final_reply = "亲，灵豆需要配合具体的订单才能帮您查询物流状态哦~ 如果您遇到售后或退换货问题，请联系人工客服帮您妥善处理哦！"
        
    elif intent_res.intent == "aftersale":
        # 拦截：精细化售后工单逻辑
        sub_intent = intent_res.sub_intent
        logger.info(f"[{openid}] Routing to AFTERSALE handler. Sub-intent: {sub_intent}")
        
        # 确定订单
        def _is_valid_order(oid):
            if not oid: return False
            return str(oid).lower().strip() not in ["null", "none", "", "无", "unknown"]
            
        extracted_order_id = intent_res.order_id if _is_valid_order(intent_res.order_id) else (request.order_id if _is_valid_order(request.order_id) else None)
        
        # 预警标签
        tag_map = {
            "damage": "🚨破损/少件/瑕疵",
            "wrong": "📦发错货",
            "refund": "💸退款/七天无理由",
            "urge": "🔥催单/停滞",
            "other": "🔧其他售后诉求"
        }
        issue_tag = tag_map.get(sub_intent, "🔧未知售后")
        
        # --- 提取真实诉求原话 ---
        # 原始化抽取：防止此时用户只发了一个孤零零的订单号，我们去历史记录里捞上一条原话
        actual_complaint = query.strip()
        if extracted_order_id and (len(actual_complaint) < 10 or "订单号" in actual_complaint or extracted_order_id in actual_complaint):
            user_lines = [line.replace("用户: ", "").strip() for line in history_text.split('\n') if line.startswith("用户: ")]
            if len(user_lines) >= 2:
                # user_lines[-1] 是当前刚发送的订单号，user_lines[-2] 是上一轮真正发牢骚的话
                actual_complaint = user_lines[-2]
                
        # TODO: 转人工时，使用 LLM 智能归纳的诉求给人工看的
        # actual_complaint = intent_res.complaint_summary if intent_res.complaint_summary else actual_complaint
        
        actual_complaint_short = actual_complaint[:50]
        
        # 返回到response
        if extracted_order_id:
            log_msg = f"【商户工单推送模拟】分类: [{issue_tag}] | 用户ID: {openid} | 订单: {extracted_order_id} | 诉求原话: {actual_complaint_short} | 提醒: 请商户及时核实处理。"
            logger.info(log_msg)
            final_issue_tag = log_msg
        else:
            log_msg = f"【预警】分类: [{issue_tag}] | 用户ID: {openid} | 订单: 暂未提供 | 提醒介入安抚"
            logger.info(log_msg)
            final_issue_tag = log_msg

        base_reply = intent_res.reply.strip() if intent_res.reply else "亲，非常抱歉给您带来了不好的体验。"
        
        if extracted_order_id:
            self_service_guide = "如果您需要【退换货】，您可以快捷自助操作：打开【微信】->底部【我】->【订单与卡包】->进入找到对应订单->点击【申请售后】->【退货/退款】即可。"
            final_reply = f"{base_reply} {self_service_guide} 对于投诉或其他复杂问题，我已经为您加急转交人工专员，请亲亲耐心等待处理哦~"
        else:
            final_reply = f"{base_reply} 为了帮您进一步核实情况，请提供一下相关的订单号。如果是退换货问题，您也可以直接在【微信】->【我】->【订单与卡包】里找到该订单，进入后快捷自助申请售后哦~"
        
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
                lingdou_conv_id = await get_or_create_lingdou_conversation(openid, business_id)
                
                payload = {
                    "query": query,
                    "business_id": business_id,
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
        images=extracted_images,
        issue_tag=final_issue_tag
    )

if __name__ == "__main__":
    import uvicorn
    # Run Gateway on a different port than LingDou (8008)
    print("Starting WeChat Shop Intent Gateway on port 8010...")
    uvicorn.run(app, host="0.0.0.0", port=8010)
