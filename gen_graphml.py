import pipmaster as pm
import random
import networkx as nx
G = nx.read_graphml("rag_storage_furniture/graph_chunk_entity_relation.graphml")
# G = nx.read_graphml(r"D:\AppData\WechatData\xwechat_files\wxid_xirekto8ssfm22_4790\msg\file\2025-08\graph_chunk_entity_relation.graphml")


from pyvis.network import Network
net = Network(height="100vh", notebook=True)
net.from_nx(G)
for node in net.nodes:
    node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))  # 随机颜色
    if "description" in node:
        node["title"] = node["description"]  # 鼠标悬停显示描述
for edge in net.edges:
    if "description" in edge:
        edge["title"] = edge["description"]  # 鼠标悬停显示边描述

net.show("knowledge_graph.html")

