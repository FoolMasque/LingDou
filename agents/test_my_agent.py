import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import logging
from agents.my_agent import run_ecommerce_agent

# 配置日志查看控制台输出
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def main():
    print("=" * 60)
    print("[START] LingDou LangGraph Customer Service Agent Test (模拟非订单页入口)")
    print("=" * 60)

    session_id = "test_user_non_order_page"

    # 测试用例 1：闲聊打招呼
    q1 = "你好呀！"
    print(f"\n【买家输入 1】: {q1}")
    res1 = await run_ecommerce_agent(query=q1, conversation_id=session_id)
    print(f"识别意图: {res1['intent']} (sub_intent: {res1['sub_intent']})")
    print(f"客服回复: {res1['reply']}")

    # 测试用例 2：发起售后诉求（模拟非订单页进入，第一轮未提供订单号）
    q2 = "我买的玉米收到了但是坏了，收到包装鼓鼓的，有点漏气"
    print(f"\n【买家输入 2 (未带订单号)】: {q2}")
    res2 = await run_ecommerce_agent(query=q2, conversation_id=session_id)
    print(f"识别意图: {res2['intent']} (sub_intent: {res2['sub_intent']})")
    print(f"当前订单号: {res2['order_id']}")
    print(f"客服回复:\n{res2['reply']}")

    # 测试用例 3：买家手动回复订单号（第二轮手动补充发送订单号）
    q3 = "订单号 3734331138832749312"
    print(f"\n【买家输入 3 (补充发送订单号)】: {q3}")
    res3 = await run_ecommerce_agent(query=q3, conversation_id=session_id)
    print(f"识别意图: {res3['intent']} (sub_intent: {res3['sub_intent']})")
    print(f"已捕获订单号: {res3['order_id']}")
    print(f"客服回复:\n{res3['reply']}")

    # 测试用例 4：多轮对话追问物流（第三轮追问派送情况，无需重复发送订单号）
    q4 = "帮我查一下现在的物流派送情况"
    print(f"\n【买家输入 4 (继承订单号追问物流)】: {q4}")
    res4 = await run_ecommerce_agent(query=q4, conversation_id=session_id)
    print(f"识别意图: {res4['intent']} (sub_intent: {res4['sub_intent']})")
    print(f"继承订单号: {res4['order_id']}")
    print(f"客服回复:\n{res4['reply']}")
    
    # 测试用例 5：商品咨询（第一轮）
    q5_1 = "你们家有什么红糖推荐吗？"
    print(f"\n【买家输入 5-1】: {q5_1}")
    res5_1 = await run_ecommerce_agent(query=q5_1, user_id=session_id, business_id="wechat_shop")
    print(f"识别意图: {res5_1['intent']} (sub_intent: {res5_1['sub_intent']})")
    print(f"LingDou 会话 UUID: {res5_1['conversation_id']}")
    print(f"客服回复:\n{res5_1['reply']}")

    # 测试用例 4：商品咨询多轮续期（第二轮追问，继承上一轮同一个 user_id 自动映射会话）
    q5_2 = "那你推荐的第一个是什么包装的，好存储吗？"
    print(f"\n【买家输入 5-2】: {q5_2}")
    res5_2 = await run_ecommerce_agent(query=q5_2, user_id=session_id, business_id="wechat_shop")
    print(f"识别意图: {res5_2['intent']} (sub_intent: {res5_2['sub_intent']})")
    print(f"LingDou 会话 UUID: {res5_2['conversation_id']}")
    print(f"客服回复:\n{res5_2['reply']}")

    print("\n" + "=" * 60)
    print("✅ 测试流程完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
