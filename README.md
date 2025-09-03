# 灵豆多模态智能问答
基于RAG-Anything 1.2.7 + LightRAG构建的中文多模态智能问答系统，专为家具、电器等垂直领域设计。

## 系统概述
灵豆是个多租户RAG系统，支持处理结构化爬虫数据和企业文档，提供智能的中文多模态问答服务。系统采用知识图谱+向量检索的混合架构，特别优化了中文内容理解和图像分析能力。

### 核心特性

- **多模态处理**：同时理解文本、图片内容，提供丰富的产品推荐
- **多业务隔离**：支持家具、电器等多个垂直领域，数据完全隔离
- **中文优化**：针对中文内容和家具领域定制的提示词系统
- **高性能架构**：异步处理+并发优化，支持大规模数据处理
- **灵活扩展**：基于配置驱动，新增业务线成本极低

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      API接口层                               │
│     FastAPI + RESTful + Dify兼容接口(可能不需要） + 静态文件服务  │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                      业务处理层                              │
│   CoreSystem + BusinessConfig + 多租户管理 + 并发控制         │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                      数据适配层                               │
│   ImageManager + MultiModalProcessor + URL路径管理器          │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                      核心引擎层                              │
│   ProductionRAGInstance + LightRAG + 中文提示词系统           │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                      存储服务层                              │
│   向量数据库 + 知识图谱存储 + 本地文件系统 + 图片CDN            │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件说明

#### 1. API接口层
- **FastAPI服务器**：提供RESTful API，支持CORS和异步处理
- **静态文件服务**：图片CDN功能，支持本地图片访问
- **Dify兼容接口**：标准的外部知识库接口

#### 2. 业务处理层
- **ProductionCoreSystem**：系统总控制器，管理多业务实例
- **BusinessConfig**：业务配置管理，支持动态添加新业务
- **并发控制**：Semaphore机制控制资源使用

#### 3. 数据适配层
- **ImageManager**：批量图片下载、本地存储、URL映射
- **MultiModalProcessor**：多模态内容构建和格式转换
- **PathManager**：本地路径与远程URL的映射管理

#### 4. 核心引擎层
- **ProductionRAGInstance**：单业务RAG实例，封装LightRAG
- **中文提示词系统**：针对家具领域优化的中文Prompt
- **性能优化**：共享OpenAI客户端，并发图片处理

#### 5. 存储服务层
- **LightRAG存储**：向量数据库 + 知识图谱持久化
- **图片本地存储**：按业务分类的文件系统存储
- **配置文件**：JSON格式的系统和业务配置


### ⏳ 待实现功能

1. **企业文档处理**
   - [ ] PDF、Word、Excel文档解析
   - [ ] 增量更新机制
   - [ ] 文档分类和标签系统

2. **性能进一步优化**
   - [ ] 缓存机制（查询结果缓存）
   - [ ] 批处理优化（减少API调用）

3. **多业务扩展**
   - [ ] 马桶等垂直领域
   - [ ] 业务间数据共享机制

4. **高级查询功能**
   - [ ] 多轮对话支持
   - [ ] 用户画像和个性化推荐
   - [ ] 搜索结果排序优化

5. **接口拓展**
   - [ ] 网站集成
   - [ ] 微信智能客服


## 技术栈

### 后端框架
- **FastAPI** - 高性能异步Web框架
- **LightRAG** - 轻量级RAG框架
- **RAG-Anything 1.2.7** - 多模态RAG引擎
- 
- ### 前端框架（页面加在图文通里，端口对接清楚即可）
- **React** - 
- **typeScript** -

### 数据存储
- **向量数据库** - 基于LightRAG内置存储
- **本地文件系统** - 图片和配置存储
- **JSON配置** - 系统配置管理

### 开发工具
- **aiohttp** - 异步HTTP客户端
- **Pydantic** - 数据验证和序列化
- **asyncio** - 异步编程框架

## 快速开始

### 安装部署

1. **克隆项目**
```bash
git clone https://gitee.com/FoolMasque/LingDou.git
cd LingDou
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置系统**
```bash
# 编辑配置文件
vim config.json
```


4. **启动服务**
```bash
# 开发模式
python api/server.py

# 生产模式
#gunicorn api.server:app -w 4 -k uvicorn.workers.UvicornWorker
deploy.sh
```

### 基本使用

1. **处理爬虫数据**
```bash
curl -X POST "http://localhost:8008/api/process_data" \
  -H "Content-Type: application/json" \
  -d '{"business_id": "furniture", "json_file": "data/furniture.json"}'
```

2. **智能查询**
```bash
curl -X POST "http://localhost:8008/api/query" \
  -H "Content-Type: application/json" \
  -d '{"business_id": "furniture", "query": "推荐一个侘寂风茶桌", "mode": "hybrid"}'
```

3. **查看系统状态**
```bash
curl "http://localhost:8008/api/status/furniture"
```

## 项目结构

```
LingDou-rag-system/
├── api/                        # API接口层
│   ├── models.py              # 数据模型定义
│   ├── routes.py              # 路由处理
│   └── server.py              # FastAPI服务器
├── core/                       # 核心业务层
│   ├── components.py          # 核心组件
│   ├── rag_instance.py        # RAG实例
│   └── system.py              # 系统控制器
├── config/                     # 配置管理
│   ├── settings.py            # 配置类
│   └── prompts.py             # 中文提示词
├── utils/                      # 工具函数
│   ├── url_helper.py          # URL处理工具
│   └── logger.py              # 日志系统
├── static/                     # 静态文件存储
│   └── images/                # 图片文件夹
│       └── furniture/         # 按业务分类
├── logs/                       # 日志文件
├── rag_storage_*/             # RAG持久化存储
├── config.json                # 系统配置
├── requirements.txt           # Python依赖
└── README.md                  # 项目文档
```

## 配置说明

### 查询模式说明
- **hybrid**: 混合检索（向量+图谱）- 默认
- **local**: 局部上下文查询  
- **global**: 全局知识查询
- **naive**: 基础相似性查询


