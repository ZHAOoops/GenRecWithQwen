# GenRecWithQwen
欢迎你来到GenRecWithQwen！
GenRecWithQwen是 新手友好的基于 Qwen2 的生成式推荐系统，通过大模型理解用户偏好生成候选物品，融合 TF-IDF 关键词召回、热门物品召回构建多源策略，平衡相关性与多样性。采用 LightGBM 排序模型精准打分，内置召回率、NDCG 等评估指标量化效果。通过 Flask 封装 API，支持 HTTP 调用与缓存机制，模块化设计便于扩展。低算力需求，部署超简单。

## ✨ 核心特性

- **生成式召回**：利用Qwen2大模型理解用户偏好，生成符合需求的候选物品
- **多源融合策略**：融合关键词召回（TF-IDF）、LLM生成召回和热门物品召回，平衡相关性与多样性
- **高效排序模型**：基于LightGBM的排序层，对候选物品精准打分，提升推荐质量
- **完整评估体系**：内置召回率、准确率、NDCG等指标评估，支持模型效果量化分析
- **即开即用服务**：通过Flask封装API接口，支持HTTP调用和缓存机制，部署简单
- **模块化设计**：数据处理、召回、排序等模块解耦，便于替换或扩展新算法

## 🏗️ 系统架构

1. **数据层**  
   - 输入：用户数据（年龄、性别等）、物品数据（标题、类别等）、交互数据（评分、点击等）
   - 输出：清洗后的结构化数据，用于后续模型训练和推荐

2. **预处理层**  
   - 数据清洗：处理缺失值、标准化文本（如电影标题去年份、分词）
   - 特征工程：构建用户偏好向量（如喜欢的电影类型）、物品特征向量（如类型、关键词）

3. **召回层**  
   - 关键词召回：基于TF-IDF计算用户偏好与物品的文本相似度
   - LLM召回：通过Qwen2生成推荐候选，再匹配到实际物品库
   - 热门召回：补充高热度物品，保证新用户/长尾物品的覆盖率

4. **排序层**  
   - 输入：召回层的候选物品列表
   - 处理：LightGBM模型学习用户-物品交互模式，对候选物品打分
   - 输出：按得分排序的最终推荐列表

5. **服务层**  
   - API接口：提供推荐查询、缓存清理、健康检查等功能
   - 缓存机制：减少重复计算，提升响应速度

## 🚀 快速开始
### 环境要求

| 依赖项         | 版本要求       | 说明                     |
|----------------|----------------|--------------------------|
| Python         | 3.8+           | 编程语言                 |
| pandas         | 1.3.0+         | 数据处理                 |
| numpy          | 1.21.0+        | 数值计算                 |
| scikit-learn   | 1.0.0+         | 特征工程与相似度计算     |
| lightgbm       | 3.3.0+         | 排序模型                 |
| flask          | 2.0.0+         | API服务框架              |
| nltk           | 3.7+           | 文本处理（停用词、分词） |
| ollama         | 0.1.30+        | 本地大模型运行环境       |

### 运行
#### 先输入
mkdir ollama_recommender
cd ollama_recommender
mkdir data models kv_store

在命令行安装requirements.txt：
pip install -r requirements.txt

## 📂 代码结构
```
GenRecWithQwen/
├── data/                  # 数据集目录（需手动创建）
│   ├── ratings_processed.csv        # 用户-物品评分数据
│   ├── movies_processed.csv         # 物品（电影）信息
│   └── users_processed.csv          # 用户信息
├── data_preparation.py    # 数据加载、清洗、特征构建
├── ollama_client.py       # Ollama模型调用封装（生成推荐文本）
├── recall_modules.py      # 召回模块
│   ├── 关键词召回（TF-IDF）
│   ├── LLM召回（Qwen2生成+匹配）
│   └── 热门召回（基于评分热度）
├── ranking_module.py      # 排序模块（LightGBM模型）
├── kv_store.py            # 缓存系统（存储推荐结果，减少重复计算）
├── recommender.py         # 主程序（初始化+API服务）
├── evaluate.py            # 评估脚本（计算召回率、NDCG等指标）
├── requirements.txt       # 依赖列表
└── README.md              # 项目说明（本文档）
```

## 运行结果
### 客户端封装后运行结果示例
=== 待检查模型 ===
qwen2:7b
测试推荐结果:
生成的提示词： Recommend 5 movies for a middle-aged male who likes Drama, Comedy, Action movies.
    Only return the movie titles, one per line, without any additional text or numbering.
Ollama生成完成，耗时: 44.35秒
1. The Godfather
2. Pee-wee's Big Adventure
3. Die Hard
4. Forrest Gump
5. Inception

### 评估完成后运行结果示例
```
===== 评估结果 =====
平均召回率@10: XXXXX    # 推荐结果覆盖用户喜欢物品的比例
平均准确率@10: XXXXX    # 推荐结果中用户喜欢物品的比例
平均NDCG@10: XXXXX      # 排序质量（值越高越好，最大为1）
平均响应时间: XXXXX秒   # 单次推荐耗时
```
