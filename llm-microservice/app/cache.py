import redis
import json
from app.utils import log

redis_client = redis.Redis(
    host="127.0.0.1",
    port=6379,
    db=0,
    decode_responses=True
)

def cache_get(key: str):
    try:
        return redis_client.get(key)
    except Exception as e:
        log.error(f"[Cache GET] {e}")
        return None

def cache_set(key: str, value, ttl=600):
    """Cache for 10 minutes by default."""
    try:
        redis_client.set(key, json.dumps(value), ex=ttl)
    except Exception as e:
        log.error(f"[Cache SET] {e}")
