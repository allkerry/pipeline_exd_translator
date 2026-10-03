import os
import tiktoken

_ENCODING_NAME = "r50k_base"

# Fail-fast: кодировка грузится при импорте модуля. Если файла нет в кэше
# (TIKTOKEN_CACHE_DIR, см. Dockerfile) и сети тоже нет — контейнер падает на
# старте, а не молча пропускает все тексты без проверки лимита токенов.
_encoding = tiktoken.get_encoding(_ENCODING_NAME)


def evaluate_token_count(item_content_string: str, encoding_name: str = None) -> int:
    if item_content_string is None or len(item_content_string) <= 1:
        return 0
    encoding = _encoding if encoding_name is None else tiktoken.get_encoding(encoding_name)
    # Исключения НЕ глушим: item с ошибкой подсчёта должен отбрасываться
    # (счётчик errors в upipe), а не проходить дальше с n_tokens=0.
    return len(encoding.encode(item_content_string))


MAX_MODEL_TOKENS = int(os.getenv("MAX_MODEL_TOKENS", "400"))
