# 灵豆多模态智能问答
基于RAG-Anything + LightRAG构建的中文多模态智能问答系统，专为家具、电器等垂直领域设计。

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


### ✅ 已实现功能

1. **企业文档处理**
   - [x] PDF、Word、Excel文档解析（基于RAG-Anything + MinerU）
   - [x] 智能解析方式选择（自动选择txt/mineru）
   - [x] 动态业务扩展（运行时自动创建新业务）
   - [x] 增量更新机制
   - [ ] 文档分类和标签系统

2. **性能优化**
   - [x] 智能解析方式选择（文本型PDF使用快速txt模式）
   - [x] 知识图谱抽取可配置（entity_extract_rounds=0关闭KG）
   - [x] 缓存机制（查询结果缓存）

3. **多业务扩展与隔离**
   - [x] 动态业务创建（API调用时自动创建）
   - [x] 手动业务注册API（`/api/business/register`）
   - [x] 多租户网关架构（`/webhook/{business_id}/{app_id}`）
   - [x] 业务独立配置（如自定义输出指令、图文识别字段映射等）

4. **微信小店集成与意图识别**
   - [x] 独立连接器：微信小店 API 包装器及独立逻辑
   - [x] 智能意图分类分流：订单查询、商品咨询和售后维权分流处理
   - [x] 售后流定制化：前置索要凭证并在后台结构化推送，及针对用户的情绪安抚（一刀切安抚策略）

5. **会话数据管理**
   - [x] 会话流式输出（SSE结构化处理：包含类型、元数据等信息）
   - [x] 动态更新用户画像（`/api/conversations/{conversation_id}/metadata`）
   - [x] 会话数据清理与导出机制

6. **高级查询功能**
   - [ ] 搜索结果排序优化

7. **接口拓展**
   - [ ] 网站集成
   - [x] 微信智能客服


## 技术栈

### 后端框架
- **FastAPI** - 高性能异步Web框架
- **LightRAG** - 轻量级RAG框架
- **RAG-Anything** - 多模态RAG引擎
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

sudo apt-get install -y libreoffice-core libreoffice-writer libreoffice-calc libreoffice-impress

sudo apt install -y fonts-wqy-zenhei fonts-wqy-microhei ttf-mscorefonts-installer
```

3. **启动Redis（会话存储）**
```bash
# 方法1：使用docker-compose（推荐）
docker-compose -f docker-compose.redis.yml up -d

# 方法2：直接使用docker命令
docker run -d \
  --name lingdou-redis \
  -p 16380:6379 \
  -v lingdou_redis_data:/data \
  --restart unless-stopped \
  redis:7.2.5-alpine \
  redis-server --appendonly yes

# 验证Redis连接
redis-cli -p 16380 ping
# 应该返回: PONG
```

4. **配置系统**
```bash
# 编辑配置文件
vim config.json

# Redis已自动配置为 localhost:16380（独立Redis容器）
# 如需修改，编辑 config.json 中的 conversation.storage_config.redis.url
```


4. **启动服务**
```bash
# 开发模式
python api/server.py

# 生产模式
deploy.sh 
          start        启动服务 (默认)
          stop         停止服务
          restart      重启服务
          status       查看状态
          logs [n]     查看最近n行日志 (默认50行)
          logs live    查看实时日志
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

**新增调试断点（Debug）参数**:
- `only_need_context`: (bool) 纯资料检索模式。如果设为 `true`，系统会跳过底层大语言模型的生成阶段，只需很少的时间即可将引擎中召回的碎文本片段、知识节点以纯文本格式返回。非常适合用来测试“资料搜不搜得到 / 搜得准不准”（0 Token耗费）。
- `only_need_prompt`: (bool) 考卷全貌模式。当你想知道“我设定的人物背景和大纲真的生效了吗？大语言模型看到的实际Prompt到底长什么样？”的时候可开启此项。开启后不调取大模型，直接返回由底层系统组合好的数千字巨大指令段落（前提 `only_need_context=false`）。

3. **查看系统状态**
```bash
curl "http://localhost:8008/api/status/furniture"
```

## 项目结构

```
LingDou-rag-system/
├── api/                       # API接口层
│   ├── models.py              # 数据模型定义
│   ├── routes.py              # 路由处理
│   └── server.py              # FastAPI服务器
├── core/                      # 核心业务层
│   ├── components.py          # 核心组件
│   ├── rag_instance.py        # RAG实例
│   ├── conversation_manager.py# 会话管理
│   ├── image_processor.py     # 图片处理器
│   └── system.py              # 系统控制器
├── config/                    # 配置管理
│   ├── settings.py            # 配置类
│   └── runtime_prompt_patch.py# 运行时内存替换中文提示词
├── utils/                     # 工具函数
│   ├── url_helper.py          # URL处理工具
│   └── logger.py              # 日志系统
├── static/                    # 静态文件存储
│   └── images/                # 图片文件夹
│       └── furniture/         # 按业务分类
├── conversation/              # File模式 会话存储文件夹
│   └── *.json/                # 会话id存储
├── logs/                      # 日志文件
├── rag_storage_*/             # RAG持久化存储
├── config.json                # 系统配置
├── requirements.txt           # Python依赖
├── deploy.sh                  # 启动脚本
└── README.md                  # 项目文档
```

## 配置说明

### 查询模式 (`mode`) 说明
这 6 个词代表了底层的 RAG 搜索引擎寻找答案的策略：
- **hybrid** (混合双打模式 - 推荐默认)：结合局部与全局检索，不仅找实体还提炼网状关联，适用 90% 以上场景，效果最均衡。
- **local** (局部微看模式)：围绕提问中的具体实体（比如针对某个疾病、某个人）顺藤摸瓜检索，回答细节最深究。
- **global** (全局宏观模式)：跳出具体点，沿着跨度更大的图谱网络提取全局社区的宏观总结、适用人群、顶层架构等知识片段。
- **naive** (大力出奇迹模式)：基于传统 Vector 向量相似度直接全文检索，速度快，适合原文找句和极为固定的名词提取（无逻辑关系网络）。
- **mix** (融合战士)：`hybrid`（知识图谱检索结果）与原始 `naive`（纯文本向量）的大融合搜索。能捕捉最多最全的语料，容错率最高但算力和上下文消耗极大。
- **bypass** (逃避模式)：底层知识框架保留字，常规非特殊接口业务不建议使用。

### 文档处理配置
- `rag_anything.parse_method`: 默认 `auto`，结合 `smart_parse` 自动决策 `txt` 或 `mineru`。
- `rag_anything.smart_parse`: 开启后会对 PDF 进行结构采样，文本型 PDF 优先走极速 `txt` 管线（采样统计页数/文本密度/图片占比）。
- `rag_anything.max_txt_file_mb` / `max_txt_pages` / `image_page_ratio_threshold`: 控制“文本型 PDF”判定阈值，满足条件时走 `txt` 直读。
- `rag_anything.enable_image_processing` / `enable_table` / `enable_formula`：控制 MinerU 是否抽取图片、表格和公式内容。
- `rag_anything.entity_extract_rounds`: 控制 LightRAG 实体抽取轮次，配合 `kg_extraction_mode`（`adaptive`/`ratio`/`limit`/`all`）可在性能与图谱质量间平衡。
- `rag_anything.multimodal_weights`: 为图片/表格/公式/纯文本分配不同的图谱权重，帮助 LightRAG 更关注关键模态。

### 文档处理流程

1. **文件上传 & 业务绑定**
   - `/api/ingest/document` 在收到文件后会自动注册缺失的业务实例，确保不同业务互不干扰。
   - 支持的主流格式：`PDF`、`DOCX`、`PPTX`、`XLSX`、`TXT/MD`，都通过 RAG-Anything 的统一入口处理。

2. **智能解析策略（统一使用RAG-Anything）**
   - **所有文档类型**都通过 RAG-Anything 的统一入口处理，保证配置和流程的一致性。
   - **文本主导 PDF**：`smart_parse` 会采样多页，满足"页数≤阈值 + 文件≤阈值 + 平均字符数≥阈值 + 图片占比≤阈值"时，自动使用 `txt` 模式（pypdf直接提取），速度最快。
   - **结构复杂 PDF / Office 文件**：自动切换到 MinerU，多模态 OCR + 布局解析，能还原复杂排版、表格和公式。
   - **纯文本文件（`txt`/`md`）**：自动使用 `txt` 模式，RAG-Anything 直接读取文本内容，不进行PDF转换，复用统一的切片和知识图谱抽取逻辑。

3. **MinerU 多模态解析**
   - 同步产出：结构化文本块、表格 JSON、图片切片、公式描述，底层由 `enable_image_processing / enable_table / enable_formula` 控制。
   - 表格会拆为结构化 HTML/JSON，图片会保存原图 + 生成描述（caption），公式会以 LaTeX/文本形式回传。

4. **LightRAG 入库 & 图谱抽取**
   - RAG-Anything 将解析好的 chunk 直接写入 LightRAG，沿用统一的切片大小、重叠参数。
   - 知识图谱阶段根据 `entity_extract_rounds + kg_extraction_mode` 做自适应抽取：大文档可限制比例/数量或设置最大抽取时间，小文档可全量抽取。
   - 抽取结果（实体、关系、模态描述）会写入 LightRAG 的向量库、KV 存储与图数据库。

5. **图片路径与静态资源**
   - 解析出的图片统一迁移到 `static/images/{business_id}/`，并通过 `PathManager` 注册 URL 映射，确保查询结果里的图片路径可直接对外访问。
   - 文档内图片、表格截图、公式截图会同步登记到 chunk 内容中，后续回答可直接引用。

6. **查询阶段**
   - `query` / `aquery_multimodal_stream` 会自动识别 chunk 中的图片路径，按需将图像转为 Base64 送入视觉模型，从而实现"文本+图片"联合回答。

### URL路径后处理机制

系统实现了智能的URL路径后处理机制（`post_process_response_urls`），用于将LLM响应中的本地图片路径转换为可访问的远程URL，并防止多轮对话中的URL重复拼接问题。

#### 工作原理

1. **URL保护机制**
   - 在处理本地路径之前，先提取所有已经是完整URL的图片路径（匹配 `http://` 或 `https://` 开头的图片URL）
   - 使用占位符临时替换这些已存在的URL，避免被误处理

2. **本地路径转换**
   - 支持多种本地路径格式：
     - `static/images/{business_id}/{filename}`
     - `rag_storage_{business_id}/parsed/{doc}/images/{filename}`
     - `rag_storage_{business_id}/images/{filename}`
   - 通过 `PathManager` 查找路径映射，或使用兜底逻辑构建远程URL
   - 使用正则表达式和负向后顾断言，确保不匹配URL中的路径部分

3. **URL恢复**
   - 处理完成后，将被保护的URL从占位符恢复为原始URL
   - 确保已转换的URL不会被重复处理

#### 解决的问题

- **单轮对话**：将chunk中的本地路径（如 `rag_storage_furniture/images/xxx.jpg`）转换为远程URL（如 `http://47.100.14.93:8008/images/rag_storage_furniture/images/xxx.jpg`）
- **多轮对话**：防止历史记录中已转换的URL被重复处理，避免出现 `http://.../images/http://.../images/...` 的重复拼接问题
- **路径兼容**：支持Windows和Linux路径格式，自动统一为URL友好的正斜杠格式

#### 实现细节

```python
# 1. 保护已存在的URL
url_pattern = re.compile(r'https?://[^\s)\]]+\.(?:jpg|jpeg|png|gif|bmp|webp)')
# 用占位符替换所有已存在的URL

# 2. 匹配和转换本地路径
pattern = re.compile(r'(?<!http://)(?<!https://)(rag_storage_.../images/...)')
# 使用负向后顾断言确保不匹配URL中的路径

# 3. 恢复被保护的URL
# 将占位符恢复为原始URL
```

#### 使用场景

- **查询响应处理**：在 `rag_instance.aquery_with_history` 中自动调用，处理LLM返回的文本
- **多轮对话**：确保历史记录中的URL不会被重复转换
- **图片展示**：前端可以直接使用转换后的URL显示图片

## API接口

### 文档录入接口

#### 1. 单文档上传（支持动态业务创建）
```bash
POST /api/ingest/document
Content-Type: multipart/form-data

参数:
- file: 文档文件（PDF/DOCX/PPTX/XLSX等）
- business_id: 业务ID（如果不存在会自动创建）
- doc_type: 文档类型（默认"manual"）
- use_gpu: 是否使用GPU（默认false）
```

**特性**：
- ✅ 自动创建新业务：如果`business_id`不存在，系统会自动创建默认配置
- ✅ 智能解析：根据文档特征自动选择最优解析方式（txt/mineru）
- ✅ 性能优化：可通过`entity_extract_rounds=0`关闭知识图谱抽取，大幅提升速度

**示例**：
```bash
# 上传论文到新业务"paper"（会自动创建）
curl -X POST "http://localhost:8008/api/ingest/document" \
  -F "file=@paper.pdf" \
  -F "business_id=paper" \
  -F "use_gpu=false"
```

#### 2. 手动注册业务（可选）
```bash
POST /api/business/register
Content-Type: multipart/form-data

参数:
- business_id: 业务ID（必需）
- name: 业务名称（可选）
- image_fields: 图片字段列表，JSON或逗号分隔（可选）
- text_fields: 文本字段列表，JSON或逗号分隔（可选）
```

**示例**：
```bash
# 注册论文业务
curl -X POST "http://localhost:8008/api/business/register" \
  -F "business_id=paper" \
  -F "name=学术论文" \
  -F "text_fields=title,abstract,keywords,content"
```

### 会话管理接口

#### 1.创建会话
```http
POST /api/conversations/new?business_id=toilet&user_id=lyq HTTP/1.1
Host: 47.100.14.93:8008
```

#### 2. 动态更新会话元数据（例如用户画像）
```bash
PUT /api/conversations/{conversation_id}/metadata
Content-Type: application/json
```

**参数**:
- 请求体为 JSON 对象，包含需要更新的元数据键值对。
- 例如，使用 `user_persona` 字段来注入用户动态画像，系统会在后续对话中自动将该画像注入 LLM 的系统提示词中。

**示例**：
```bash
curl -X PUT "http://localhost:8008/api/conversations/test_conv_123/metadata" \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": "我是一名30岁的程序员，最近经常加班，有点颈椎痛。请在回答时注意这些背景。"
  }'
```

#### 3. 导出会话记录
```bash
GET /api/conversations/{conversation_id}/export?format=json
```

**参数**:
- `format`: 导出的格式（`json` 或 `text`），默认为 `json`。

#### 4. 清理旧会话
```bash
POST /api/conversations/cleanup?days=7
```

**参数**:
- `days`: (int) 保留的最近天数，早于该天数的会话将被随时清理（默认为7）。
- 由于清理逻辑可能耗时，接口设计为异步处理（使用 FastAPI 后台任务执行）。


### 业务管理接口

#### 1. 删除业务
```bash
DELETE /api/businesses/{business_id}
```

**特性**：
- ✅ 彻底删除：移除业务配置、RAG实例及相关数据目录
- ⚠️ 危险操作：数据不可恢复，请谨慎使用

**示例**：
```bash
curl -X DELETE "http://localhost:8008/api/businesses/paper"
```

#### 2. 更新业务配置
```bash
PUT /api/businesses/{business_id}/config
Content-Type: application/json
```

**参数**:
- `response_instruction`: (str) 自定义回复指导，例如"请用海盗语气回答"
- `field_mapping`: (dict) 字段映射，例如 `{"product_name": "item_name"}`
- `vision_prompt_template`: (str) 自定义视觉分析提示词

**示例**：
```bash
curl -X PUT "http://localhost:8008/api/businesses/furniture/config" \
  -H "Content-Type: application/json" \
  -d '{
    "response_instruction": "请用简短的列表形式回答",
    "field_mapping": {
        "product_name": "item_title",
        "product_image": "main_pic"
    }
  }'
```

### 文档录入与管理接口

#### 1. 单文档与批量文档上传（支持覆盖/重置）
```text
POST /api/ingest/document
POST /api/ingest/batch_documents
```

**新增参数**:
- `overwrite`: (bool) 可选参数，默认为 `false`。
  - 如果传 `true`，系统在录入新文档前，会自动根据**上传的文件名**扫描并删除已属于该业务的旧版本文档分块与知识图谱实体，确保不存在新旧版本知识冲突的问题。

**示例**（携带 overwrite 参数）：
```bash
curl -X POST "http://localhost:8008/api/ingest/document" \
  -F "file=@/path/to/your_document.pdf" \
  -F "business_id=ARglasses" \
  -F "doc_type=manual" \
  -F "overwrite=true"
```

#### 2. 删除指定文档的 RAG 知识
```bash
DELETE /api/ingest/document/{business_id}/{doc_id}
```

**特性**:
- `doc_id` 传递文档的原始文件名（需进行 URL 编码，例如 `智能眼镜.pdf` -> `%E6%99%BA%E8%83%BD%E7%9C%BC%E9%95%9C.pdf`）。
- 内部自动进行哈希映射，底层同步销毁存在于 LightRAG 向量库与图数据库里的相关实体特征，以保证数据干净无残留。

**示例**：
```bash
curl -X DELETE "http://localhost:8008/api/ingest/document/ARglasses/test.pdf"
```
