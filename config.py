OLLAMA_MODEL = "qwen2:1.5b"
TOP_K = 10
CACHE_ENABLED = True
CACHE_TTL = 3600  # 缓存有效期1小时
LLM_TIMEOUT = 30  # LLM调用超时30秒
LLM_OPTIONS = {
    "num_predict": 50,      # 限制生成长度（关键优化）
    "temperature": 0.7,
    "top_p": 0.9,
    "num_thread": 8,        # 根据CPU核心调整
}
