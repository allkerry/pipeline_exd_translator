# Архитектура (P106-100, 6 ГБ)

```
                      ┌──────────────► translator:8003 (NLLB-1.3B, CT2 int8, GPU)
                      │ не-en
collector:9000 → upipe:5981 → bpipe:7995 → transactioneer:8002 → Exorde Upload API
 (LANG_FILTER пуст)   │ en                                           │
                      └─ без вызова translator                  vpn (Xray REALITY)
```
Порядок старта (depends_on, healthy): `vpn → translator → transactioneer → bpipe → upipe → collector`. GPU делят `translator` и `bpipe`; translator занимает VRAM первым.

## Модули
- **collector** — фильтры: длина, дубликаты, возраст, (опц.) язык; обрезка до `MAX_TEXT_LEN`.
- **upipe** — `translate.py` (детект языка → en как есть / не-en → `translator`), keywords (YAKE), лимит токенов.
- **translator** — `POST /translate`, `GET /health`. Старт: скачивание → конвертация HF→CT2 int8 (идемпотентно, `.done`) → загрузка на GPU → прогрев → `ready`.
  - Сегментация по предложениям (≤150 токенов, суммарно ≤`MAX_TRANSLATE_SRC_TOKENS`), микробатчинг (≤`TRANSLATOR_MAX_BATCH` сегментов или 15 мс), одна GPU-очередь.
  - Защита: детект повторов/раздутого вывода → повторный перевод с `repetition_penalty`; OOM → повтор по одному → 503 на запрос.
- **bpipe** — эмбеддинги, zero-shot, emotion, irony, text-type, sentiment (`TAG_BATCH_SIZE`, `BPIPE_TORCH_DTYPE`).
- **transactioneer** — загрузка батчей на Upload API через VPN.
- **vpn** — Xray-core VLESS+REALITY, SOCKS5 :1080 / HTTP :1081.

## Контракт upipe ↔ translator
```
POST /translate  {"items":[{"text":"...","src":"rus_Cyrl"}]}
→ 200 {"translations":[{"text":"...","truncated":false}|null]}
→ 503 translator не готов / ошибка GPU
```
Ошибки translator в upipe = `TranslatorError` (счётчик `errors`); неподдерживаемый язык = `NonEnglishError` (счётчик `filtered_lang`).