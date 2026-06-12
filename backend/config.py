import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Flask
    FLASK_APP = 'app.py'
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_ENV') != 'production'

    # ChromaDB
    CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_db')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'game_opinions')

    # LLM
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'deepseek')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    QWEN_API_KEY = os.getenv('QWEN_API_KEY')
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.3'))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '500'))

    # Embedding
    EMBEDDING_MODEL = 'BAAI/bge-small-zh-v1.5'
    EMBEDDING_DIMENSION = 384

    # RAG
    TOP_K = int(os.getenv('TOP_K', '15'))
    CONTEXT_MAX_LENGTH = int(os.getenv('CONTEXT_MAX_LENGTH', '4000'))

    # MySQL (Text-to-SQL)
    MYSQL_HOST = os.getenv('MYSQL_HOST', '127.0.0.1')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3306'))
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'standardized_data')

    # Hybrid search
    BM25_ENABLED = os.getenv('BM25_ENABLED', 'true').lower() == 'true'
    HYBRID_TOP_K = int(os.getenv('HYBRID_TOP_K', '20'))
    RRF_K = int(os.getenv('RRF_K', '60'))

    # Structured filter: LLM extracts ChromaDB WHERE conditions from question
    STRUCTURED_FILTER_ENABLED = os.getenv('STRUCTURED_FILTER_ENABLED', 'true').lower() == 'true'

    # Query expansion: game-domain synonyms for BM25
    QUERY_EXPANSION_ENABLED = os.getenv('QUERY_EXPANSION_ENABLED', 'true').lower() == 'true'

    # Time decay
    TIME_DECAY_ENABLED = os.getenv('TIME_DECAY_ENABLED', 'true').lower() == 'true'
    TIME_DECAY_HALF_LIFE_DAYS = int(os.getenv('TIME_DECAY_HALF_LIFE_DAYS', '30'))

    # Hot-score boost
    HOT_BOOST_ENABLED = os.getenv('HOT_BOOST_ENABLED', 'true').lower() == 'true'

    # Dedup + Diversity
    DEDUP_ENABLED = os.getenv('DEDUP_ENABLED', 'true').lower() == 'true'
    DIVERSITY_ENABLED = os.getenv('DIVERSITY_ENABLED', 'true').lower() == 'true'
    DIVERSITY_MAX_PER_GAME = int(os.getenv('DIVERSITY_MAX_PER_GAME', '3'))

    PROMPT_TEMPLATE = os.getenv('PROMPT_TEMPLATE', '''你是游戏舆情分析助手。知识库覆盖B站和NGA平台的手游玩家讨论、评论、视频，涵盖各大手游热度、口碑、争议事件。

【游戏黑话速查表】
- "节奏" = 争议事件 / 玩家集体不满 / 炎上
- "凉了" = 热度大幅下降 / 玩家流失严重
- "暴死" = 上线后口碑惨败
- "喷" / "冲" = 玩家集中批评 / 社区负面爆发
- "吃瓜" = 围观争议不站队
- "米池" = 米哈游抽卡机制(原神/星铁/绝区零)
- "长草" = 游戏内容匮乏期
- "肝" = 需要大量时间投入
- "氪" = 充值付费

【参考知识】
{context}

【用户问题】
{question}

回答规则：
1. 从参考知识提取信息，先归纳已知部分，再逐条说明数据局限
2. 提了具体游戏但无数据 -> "该游戏暂无监控数据"，不编造
3. 涉及"最近/近期"时注明每条引用数据的发布时间
4. 争议话题客观呈现多方观点
5. 3-5句话，结论 + 论据 + 局限''')
