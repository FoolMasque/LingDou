import hashlib
import re
from typing import TypedDict, List, Dict, Optional, Any
from langgraph.graph import StateGraph, END, START
import os
import asyncio
import httpx
import json
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
logger = logging.getLogger(__name__)
# 当前路径下.env
agents_env = os.path.join(os.path.dirname(__file__), ".env")
# Load environment variables if a .env file exists this folder 先这样默认根目录下还没.env
load_dotenv(dotenv_path=agents_env)

class AgentState(TypedDict):
    messages:List[Dict[str,Any]] # history
    query:str
    intent:str
    sub_intent:str
    order_id:str
    business_id:str
    conversation_id:str
    user_id:str
    final_response: str
    app_id:str


def _get_llm() -> Optional[ChatOpenAI]:
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.7
    )
    return llm


# =========节点======
async def classify_intent_node(state: AgentState) -> AgentState:
    """
    结合历史对话，提取主意图与子意图。
    """
    query = state.get("query", "").strip()
    messages = state.get("messages", [])

    # 1. 优先提取 19 位订单号存入 state，避免买家在咨询商品时带订单号而遗漏
    extracted_order_id = extract_order_from_query(query)
    if extracted_order_id:
        state["order_id"] = extracted_order_id

    # 2. 提取最近 4 条历史对话文本作为上下文
    history_lines = [f"{m.get('role')}: {m.get('content')}" for m in messages[-4:]]
    history_text = "\n".join(history_lines) if history_lines else "无"

    intent = "chat"
    sub_intent = "none"

    llm = _get_llm()
    if llm:
        try:
            # 调 Qwen 大模型识别主意图与子意图
            system_prompt = """你是一个专业的电商智能客服意图识别引擎。
请结合【历史对话】与【最新用户输入】，判断用户的核心意图，并输出 JSON。

【主意图 intent】(必须是以下之一):
1. "product": 商品咨询（问规格、价格、材质、功能、推荐、怎么买等）。
2. "order": 订单服务（包含用户单独发送订单号、物流查询、发货状态、退换货、商品破损、发错货、退款、催单等所有与订单/售后有关的问题）。
3. "chat": 闲聊问候（你好、在吗、谢谢等非业务问候）。

【订单子意图 sub_intent】(仅当 intent 为 "order" 时判断，否则填 "none"):
- "logistics": 客观询问发货/物流轨迹。
- "damage": 商品或包装破损、缺件、质量瑕疵。
- "wrong": 发错货（颜色/款式/型号不符）。
- "refund": 退货/退款/七天无理由。
- "urge": 催发货/抱怨物流太慢。
- "other": 其他订单或售后问题（包含用户发送订单号补充信息）。

输出 JSON 格式必须为:
{"intent": "product|order|chat", "sub_intent": "logistics|damage|wrong|refund|urge|other|none"}"""

            user_prompt = f"【历史对话】\n{history_text}\n\n【最新用户输入】\n{query}"

            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            res_text = response.content.strip()
            if "```" in res_text:
                res_text = res_text.split("```")[1].replace("json", "").strip()
            data = json.loads(res_text)

            intent = data.get("intent", "chat")
            sub_intent = data.get("sub_intent", "none")

        except Exception as e:
            logger.warning(f"LLM 意图识别异常，触发规则降级: {e}")
            if extracted_order_id or any(w in query for w in ["订单", "单号", "退款", "破损", "发错", "坏了", "催单", "发货", "快递", "物流"]):
                intent = "order"
                sub_intent = "logistics" if any(w in query for w in ["发货", "快递", "物流"]) else "other"
            elif any(w in query for w in ["玉米", "红糖", "品质", "价格"]):
                intent = "product"

    # 3. 更新意图状态
    state["intent"] = intent
    state["sub_intent"] = sub_intent
    return state

LINGDOU_API_URL = os.getenv("LINGDOU_API_URL", "http://localhost:8008/api/query")

async def product_node(state:AgentState):
    query = state.get("query", "").strip()
    business_id = state.get("business_id", "")
    conv_id = state.get("conversation_id", "")
    user_id = state.get("user_id", "")
    logger.info(f"[Node: Product] 调用 LingDou 接口: query='{query}', conversation_id='{conv_id}', user_id='{user_id}'")
    
    payload = {
        "business_id": business_id,
        "query": query,
        "mode": "mix",
        "streaming": False
    }
    # 只要有 LingDou 的 conversation_id UUID，就透传供 LingDou 续期
    if conv_id and conv_id != "default_session":
        payload["conversation_id"] = conv_id
    if user_id:
        payload["user_id"] = user_id

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                LINGDOU_API_URL,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=180.0
            )
            resp.raise_for_status()

            data = resp.json()
            reply = data.get("result", "抱歉，灵豆系统开小差了，请稍后再试~")
            
            # 统一提取并更新 conversation_id 为 LingDou 生成的真实会话 UUID
            returned_conv_id = data.get("conversation_id")
            if returned_conv_id:
                state["conversation_id"] = returned_conv_id
                logger.info(f"[Node: Product] 已更新/续期 LingDou 后端会话 UUID: {returned_conv_id}")

    except Exception as e:
        logger.error(f"[Node: Product] 调用 LingDou /api/query 接口失败: {e}")
        reply = "亲，灵豆的商品知识库系统有点小波动，您可以先告诉我您看中了哪款产品的尺寸或规格哦~"
    # 将回答写回 State
    state["final_response"] = reply
    return state


async def order_node(state: AgentState) -> AgentState:
    """
    订单
    """
    query = state.get("query", "").strip()
    sub_intent = state.get("sub_intent", "logistics")
    user_id = state.get("conversation_id", "guest_user")
    app_id = state.get("app_id", "wx_store_default")
    logger.info(f"[Node: Order] 处理订单主线: sub_intent='{sub_intent}', query='{query}'")

    # 1. 优先使用已有的 order_id (前端传的)，没有则调用正则从 query 里提取
    order_id = state.get("order_id") or extract_order_from_query(query)
    if order_id:
        state["order_id"] = order_id  # 存入 state 供多轮续期

    # 2. 分支一：客观物流查询 (sub_intent == "logistics")
    if sub_intent == "logistics":
        if order_id:
            # 调微信小店 API 由 19 位订单号换取快递单号 (waybill_id) 与快递公司 (delivery_name)
            waybill_id, delivery_name = await fetch_wechat_order_tracking_tool(app_id, order_id)
            
            if waybill_id:
                carrier_map = {
                    "圆通速递": "yuantong", "中通快递": "zhongtong", "京东物流": "jd",
                    "韵达快递": "yunda", "申通快递": "shentong", "极兔速递": "jtexpress",
                    "邮政电商标快": "youzhengdsbk", "德邦快递": "debangkuaidi", "菜鸟速递": "danniao",
                    "邮政标准快递": "youzhengbk", "中通快运": "zhongtongkuaiyun", "跨越速运": "kuayue",
                    "德邦物流": "debangwuliu", "京东快运": "jingdongkuaiyun", "顺丰快运": "shunfengkuaiyun",
                    "安能快运": "annengwuliu", "顺丰速运": "shunfeng"
                }
                carrier = carrier_map.get(delivery_name, "yuantong")
                
                # 调快递100 查真实轨迹
                logistics_data = await fetch_logistics_info_tool(waybill_id, carrier)

                # 调大模型结合【用户问法】与【真实物流 JSON 数据】做自然、有温度的客服提炼总结
                llm = _get_llm()
                if llm:
                    try:
                        system_prompt = """你是一个智能客服-灵豆。
                        用户正在查询订单/物流信息。
                        以下是快递 API 返回的真实物流 JSON 数据。
                        请你根据用户的具体诉求，将 JSON 里的物流轨迹提炼成一段自然、简短、有温度的客服回复。
                        切记：只能依据提供的 JSON 数据回答，绝不能自己编造任何状态或时间！如果查不到，就温柔地告知用户暂时查不到。"""

                        user_prompt = f"【用户问题】\n{query}\n\n【物流单号】\n{waybill_id}\n\n【物流数据】\n{json.dumps(logistics_data, ensure_ascii=False)}"
                        
                        response = await llm.ainvoke([
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_prompt)
                        ])
                        state["final_response"] = response.content.strip()
                    except Exception as e:
                        logger.warning(f"[Node: Order] 物流回复 LLM 生成异常，使用格式化兜底: {e}")
                        latest_track = "发货揽收中"
                        if isinstance(logistics_data, dict) and logistics_data.get("data"):
                            latest_track = logistics_data["data"][0].get("context", "派送中")
                        state["final_response"] = f"亲，为您查到订单【{order_id}】的最新物流轨迹：\n🚚 运单号: {waybill_id} ({delivery_name or '快递'})\n{latest_track}\n请您耐心等待签收哦~"
                else:
                    latest_track = "发货揽收中"
                    if isinstance(logistics_data, dict) and logistics_data.get("data"):
                        latest_track = logistics_data["data"][0].get("context", "派送中")
                    state["final_response"] = f"亲，为您查到订单【{order_id}】的最新物流轨迹：\n🚚 运单号: {waybill_id} ({delivery_name or '快递'})\n{latest_track}\n请您耐心等待签收哦~"
            else:
                # 没查到快递单号或历史单号已过期时友好提示
                state["final_response"] = f"亲，灵豆帮您查了订单（{order_id}），暂时未查询到实时物流单号。可能是商品尚未发货，或者订单较久已过快递有效查询期哦~ 如需进一步确认，您可以联系人工客服为您核实。"
        else:
            state["final_response"] = "亲，灵豆需要配合具体的订单才能帮您查询物流状态哦~ 如果您遇到售后或退换货问题，请联系人工客服帮您妥善处理哦！"

    # 3. 分支二：售后工单 SOP (sub_intent 为 damage / wrong / refund / urge / other)
    else:
        tag_map = {
            "damage": "🚨破损/少件/瑕疵",
            "wrong": "📦发错货",
            "refund": "💸退款/七天无理由",
            "urge": "🔥催单/停滞",
            "other": "🔧售后服务"
        }
        issue_tag = tag_map.get(sub_intent, "🔧售后服务")

        # 生成符合 SOP 凭证要求的亲切安抚回复
        llm = _get_llm()
        if llm:
            try:
                system_prompt = f"""你是一个温柔专业的电商售后客服“灵豆”。
                针对买家遇到的售后问题进行安抚，并按照以下 SOP 规则要求买家提供对应凭证：
                
                【售后分类】: {issue_tag}
                【凭证要求】:
                - 如果是破损(damage): 提示买家拍照提供【内外包装破损照】和【商品破损照】。
                - 如果是发错货(wrong): 提示买家拍照提供【收到的实物照片】。
                - 如果是退款(refund): 提醒买家“退换货请保证原包装完整不影响二次销售哦”。
                - 如果是催单(urge): 安抚买家“我们已加急催促快递站点优先派送”。
                
                要求：控制在 80 字以内，充满抚慰感，严禁包含联系方式。"""

                response = await llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=query)
                ])
                base_reply = response.content.strip()
            except Exception as e:
                logger.warning(f"[Node: Order] LLM 生成 SOP 失败: {e}")
                base_reply = "亲，非常抱歉给您带来了不好的体验。"
        else:
            base_reply = "亲，非常抱歉给您带来了不好的体验。"

        # 防止此时买家只发了一个孤零零的订单号，从历史记录取上一轮原话
        actual_complaint = query.strip()
        history = state.get("messages", [])
        if order_id and (len(actual_complaint) < 10 or order_id in actual_complaint):
            user_msgs = [m.get("content", "") for m in history if m.get("role") == "user"]
            if len(user_msgs) >= 2:
                actual_complaint = user_msgs[-2]

        # 工单日志与自助手续引导
        if order_id:
            log_msg = f"【商户工单推送模拟】分类: [{issue_tag}] | 用户ID: {user_id} | 订单: {order_id} | 诉求原话: {actual_complaint[:50]} | 提醒: 请商户及时核实处理。"
            logger.info(log_msg)
            self_service_guide = "如果您需要【退换货】，您可以快捷自助操作：打开【微信】->底部【我】->【订单与卡包】->进入找到对应订单->点击【申请售后】->【退货/退款】即可。"
            state["final_response"] = f"{base_reply} {self_service_guide} 对于投诉或其他复杂问题，我已经为您加急转交人工专员，请亲亲耐心等待处理哦~"
        else:
            log_msg = f"【预警】分类: [{issue_tag}] | 用户ID: {user_id} | 订单: 暂未提供 | 提醒介入安抚"
            logger.info(log_msg)
            state["final_response"] = f"{base_reply} 为了帮您进一步核实情况，请提供一下相关的订单号。如果是退换货问题，您也可以直接在【微信】->【我】->【订单与卡包】里找到该订单，进入后快捷自助申请售后哦~"

    return state


async def chat_node(state:AgentState) -> AgentState:
    """
    通用闲聊/问候分支
    """
    query = state.get("query", "").strip()
    logger.info(f"[Node: GeneralChat] 处理通用闲聊: query='{query}'")
    llm = _get_llm()
    if llm:
        try:
            system_prompt = """你是一个热情、很有礼貌的电商智能客服助手“灵豆”。
    对于用户的打招呼、问候或感谢，给予简短、温馨且专业的客服回复。
    回复中可以善意引导买家：“您可以向我咨询商品规格或者查询订单哦~”
    控制在 50 字以内，语气亲切。"""

            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ])
            reply = response.content.strip()
        except Exception as e:
            logger.warning(f"[Node: GeneralChat] LLM 异常，使用客服兜底语: {e}")
            reply = "亲，灵豆随时在线为您服务！您可以向我咨询商品规格或订单相关的哦~"
    else:
        reply = "亲，灵豆随时在线为您服务！请问有什么可以帮您？"
    # 把最终回复写入 state
    state["final_response"] = reply
    return state



# 路由跳转 根据intent跳不同sub intent分支
def route_intent(state:AgentState) -> str:
    if state["intent"] == "product":
        return "product_node" # 商品咨询
    elif state["intent"] == "order":
        return "order_node"
    else:
        return "chat_node"


# ======Tool======
def extract_order_from_query(query: str) -> Optional[str]:
    """
    优先抽取 19 位微信小店纯数字订单号，兼容匹配 SF/YT 及 6-20 位普通快递单号。
    """
    if not query:
        return None

    # 清理空格、横杠及中文冒号干扰
    clean_query = query.replace(" ", "").replace("-", "").replace("：", ":")

    # 1. 优先匹配微信小店 19 位纯数字订单号 (防中文 \b 边界失效)
    wechat_match = re.search(r'(?:^|\D)(\d{19})(?:\D|$)', clean_query)
    if wechat_match:
        return wechat_match.group(1)

    # 2. 兼容匹配普通快递单号 (6-20 位数字或 SF12345678 带字母单号)
    broad_match = re.search(r'(?:^|\D)(\d{6,20}|[A-Za-z]{1,4}\d{8,18})(?:\D|$)', clean_query)
    if broad_match:
        return broad_match.group(1)

    return None


# 微信小店根据 19 位 order_id 换取快递单号与快递公司
async def fetch_wechat_order_tracking_tool(app_id: str, order_id: str) -> tuple[Optional[str], Optional[str]]:
    """调微信小店官方 API 获取 waybill_id (快递单号) 和 delivery_name (快递公司)"""
    async with httpx.AsyncClient() as client:
        try:
            # 1 获取动态 Store Access Token (远程后端注册路径带 //)
            secret = os.getenv("WECHAT_STORE_SECRET", "1024%40Yinyu")
            token_url = f"http://47.100.14.93/backend//lingdou/wx_store/{app_id}/token?secret={secret}"
            token_resp = await client.get(token_url, timeout=10.0)
            token_resp.raise_for_status()
            token_data = token_resp.json()

            access_token = None
            if isinstance(token_data, str):
                try:
                    token_json = json.loads(token_data)
                    token_data = token_json
                except json.JSONDecodeError:
                    access_token = token_data.strip()

            if isinstance(token_data, dict):
                access_token = token_data.get("access_token") or (token_data.get("data") or {}).get("access_token")

            if not access_token:
                logger.error(f"无法获取微信店铺 access_token: {token_data}")
                return None, None

            # 2 查微信小店订单详情
            order_url = f"https://api.weixin.qq.com/channels/ec/order/get?access_token={access_token}"
            order_resp = await client.post(order_url, json={"order_id": order_id}, timeout=10.0)
            order_resp.raise_for_status()
            order_data = order_resp.json() or {}

            if order_data.get("errcode", 0) != 0:
                logger.error(f"微信小店 API 返回错误 {order_data.get('errcode')}: {order_data.get('errmsg')}")
                return None, None

            order_info = order_data.get("order") or {}
            order_detail = order_info.get("order_detail") or {}
            delivery_info = order_detail.get("delivery_info") or {}
            delivery_product_info = delivery_info.get("delivery_product_info") or []

            if delivery_product_info and len(delivery_product_info) > 0:
                pkg = delivery_product_info[0] or {}
                return pkg.get("waybill_id"), (pkg.get("delivery_name") or pkg.get("delivery_id"))
            return None, None
        except Exception as e:
            logger.error(f"查微信小店订单异常: {e}")
            return None, None


# 调快递 100 接口查真实的物流轨迹
async def fetch_logistics_info_tool(tracking_number: str, com: str = "yuantong") -> dict:
    """调 KuaiDi100 API 查询真实物流轨迹"""
    key = os.getenv("KUAIDI100_KEY", "EbZqPMOG1512")  # 授权 key
    customer = os.getenv("KUAIDI100_CUSTOMER", "7E82DA37D3BF142CABAA0F1FEEBB0374")  # 公司编号
    url = 'https://poll.kuaidi100.com/poll/query.do'
    param = {
        'com': com,
        'num': tracking_number,
        'phone': '', 'from': '', 'to': '',
        'resultv2': '1', 'show': '0', 'order': 'desc'
    }
    param_str = json.dumps(param)

    # MD5 签名算法
    sign = hashlib.md5((param_str + key + customer).encode()).hexdigest().upper()
    request_data = {'customer': customer, 'param': param_str, 'sign': sign}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, data=request_data, timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"快递100接口请求失败: {e}")
            return {"message": "网络请求失败"}





def build_graph(checkpointer: Optional[Any] = None):
    """
    构建并编译 LangGraph 电商客服 StateGraph
    """
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent_node", classify_intent_node)
    builder.add_node("product_node", product_node)
    builder.add_node("order_node", order_node)
    builder.add_node("chat_node", chat_node)

    # 设置起点入口
    builder.add_edge(START, "classify_intent_node")
    builder.add_conditional_edges("classify_intent_node",route_intent,
                                  {"product_node":"product_node",
                                   "order_node": "order_node",
                                   "chat_node": "chat_node"
                                   })
    builder.add_edge("product_node",END)
    builder.add_edge("order_node", END)
    builder.add_edge("chat_node", END)
    if checkpointer is None:
        checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)
    logger.info("✅ LangGraph 电商客服 StateGraph 编译完成！")
    return app

# 全局单例编译图应用
agent_app = build_graph()

async def run_ecommerce_agent(
    query: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    order_id: Optional[str] = None,
    business_id: str = "wechat_shop",
    app_id: str = "wx_store_default"
) -> Dict[str, Any]:
    """
    Agent 执行入口
    支持结合 MemorySaver 和 thread_id 实现多轮上下文状态恢复与结果返回
    """
    # 优先使用 user_id，或已有的 conversation_id 作为 LangGraph Checkpointer 检索的主键
    thread_id = user_id or conversation_id or "default_session"
    config = {"configurable": {"thread_id": thread_id}}

    current_state = agent_app.get_state(config)
    raw_messages = current_state.values.get("messages", []) if current_state and current_state.values else []
    messages = list(raw_messages)
    existing_order_id = current_state.values.get("order_id") if current_state and current_state.values else None
    existing_conv_id = current_state.values.get("conversation_id") if current_state and current_state.values else None
    existing_user_id = current_state.values.get("user_id") if current_state and current_state.values else None

    # 追加当前买家提问
    messages.append({"role": "user", "content": query})

    actual_conv_id = conversation_id or existing_conv_id or ""
    actual_user_id = user_id or existing_user_id or ""

    input_state: AgentState = {
        "messages": messages,
        "query": query,
        "intent": "chat",
        "sub_intent": "none",
        "order_id": order_id or existing_order_id or "",
        "business_id": business_id,
        "conversation_id": actual_conv_id,
        "user_id": actual_user_id,
        "app_id": app_id,
        "final_response": ""
    }

    # 2. 异步执行 LangGraph 状态图
    output_state = await agent_app.ainvoke(input_state, config=config)

    # 3. 追加助手回复并安全保存多轮对话历史
    history = list(output_state.get("messages", []))
    final_reply = output_state.get("final_response", "")
    final_conv_id = output_state.get("conversation_id") or actual_conv_id

    if final_reply:
        history.append({"role": "assistant", "content": final_reply})
    
    agent_app.update_state(config, {
        "messages": history,
        "conversation_id": final_conv_id,
        "user_id": actual_user_id
    })

    return {
        "reply": output_state.get("final_response"),
        "intent": output_state.get("intent"),
        "sub_intent": output_state.get("sub_intent"),
        "order_id": output_state.get("order_id"),
        "conversation_id": final_conv_id,
        "user_id": actual_user_id
    }