import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import orjson
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [translator] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("translator")

PORT = int(os.getenv("TRANSLATOR_PORT", "8003"))
MODEL_ID = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-1.3B")
CT2_DIR = os.path.join("/root/.cache", os.getenv("NLLB_CT2_DIR", "nllb-1.3b-distilled-ct2-int8"))
MAX_BATCH = int(os.getenv("TRANSLATOR_MAX_BATCH", "8"))
MAX_SRC_TOKENS = int(os.getenv("MAX_TRANSLATE_SRC_TOKENS", "600"))
BEAM = int(os.getenv("TRANSLATOR_BEAM", "4"))
SEG_TOKENS = 150
MAX_DECODE = 256
BATCH_WINDOW = 0.015
MAX_ITEMS_PER_REQUEST = 512
TGT = "eng_Latn"

NLLB_CODES = set("""
afr_Latn arb_Arab bul_Cyrl ben_Beng cat_Latn ces_Latn cym_Latn dan_Latn deu_Latn ell_Grek
spa_Latn est_Latn pes_Arab fin_Latn fra_Latn guj_Gujr heb_Hebr hin_Deva hrv_Latn hun_Latn
ind_Latn ita_Latn jpn_Jpan kan_Knda kor_Hang lit_Latn lvs_Latn mkd_Cyrl mal_Mlym mar_Deva
npi_Deva nld_Latn nob_Latn pan_Guru pol_Latn por_Latn ron_Latn rus_Cyrl slk_Latn slv_Latn
som_Latn als_Latn swe_Latn swh_Latn tam_Taml tel_Telu tha_Thai tgl_Latn tur_Latn ukr_Cyrl
urd_Arab vie_Latn zho_Hans zho_Hant
""".split())

NO_SPACE = {"zho_Hans", "zho_Hant", "jpn_Jpan"}
DENSE_SCRIPTS = {"zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang", "tha_Thai"}
REPEAT_KW = {"repetition_penalty": 1.3, "no_repeat_ngram_size": 3}

SPLIT_RE = re.compile(r"(?<=[.!?…؟।])\s+|(?<=[。！？])|\n+")

state = {"stage": "starting", "device": "cuda", "compute_type": None}
translator = None
tok = None
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ct2")
queue: asyncio.Queue | None = None
_nvml_ok = None


class GpuError(Exception):
    pass


class Seg:
    __slots__ = ("tokens", "src_len", "ratio", "fut")

    def __init__(self, tokens, ratio):
        self.tokens = tokens
        self.src_len = len(tokens) - 2
        self.ratio = ratio
        self.fut = None


def vram_used_mb():
    global _nvml_ok
    try:
        import pynvml
        if not _nvml_ok:
            pynvml.nvmlInit()
            _nvml_ok = True
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(h).used // 2**20)
    except Exception:
        return None


def _enc(text):
    return tok.encode(text, add_special_tokens=False)


def make_segments(text, src):
    sents = [s.strip() for s in SPLIT_RE.split(text) if s and s.strip()]
    units = []
    for s in sents:
        ids = _enc(s)
        if not ids:
            continue
        if len(ids) <= SEG_TOKENS:
            units.append((s, len(ids)))
        else:
            for i in range(0, len(ids), SEG_TOKENS):
                ch = ids[i:i + SEG_TOKENS]
                units.append((tok.decode(ch, skip_special_tokens=True), len(ch)))

    sep = "" if src in NO_SPACE else " "
    ratio = 6.0 if src in DENSE_SCRIPTS else 3.0
    segs, cur, cur_n, total, truncated = [], [], 0, 0, False

    def flush():
        nonlocal cur, cur_n
        if cur:
            ids = _enc(sep.join(cur))
            toks = [src] + tok.convert_ids_to_tokens(ids) + ["</s>"]
            segs.append(Seg(toks, ratio))
        cur, cur_n = [], 0

    for u, n in units:
        if total + n > MAX_SRC_TOKENS:
            truncated = True
            break
        if cur and cur_n + n > SEG_TOKENS:
            flush()
        cur.append(u)
        cur_n += n
        total += n
    flush()
    return segs, truncated


def _repeats(toks, k=4):
    n = len(toks)
    for g in range(1, 7):
        streak = 0
        for i in range(g, n):
            if toks[i] == toks[i - g]:
                streak += 1
                if streak >= (k - 1) * g:
                    return True
            else:
                streak = 0
    return False


def _bad(seg, out):
    return len(out) > max(seg.ratio * seg.src_len, 8) or _repeats(out)


def _ct2(token_lists, extra=None):
    kw = dict(
        beam_size=BEAM,
        max_decoding_length=MAX_DECODE,
        max_batch_size=MAX_BATCH,
        batch_type="examples",
        return_scores=False,
        target_prefix=[[TGT]] * len(token_lists),
    )
    if extra:
        kw.update(extra)
    res = translator.translate_batch(token_lists, **kw)
    return [r.hypotheses[0][1:] for r in res]


def _decode(toks):
    ids = tok.convert_tokens_to_ids(toks)
    return tok.decode(ids, skip_special_tokens=True).strip() or None


def _infer_segments(segs):
    outs = _ct2([s.tokens for s in segs])
    res = [None] * len(segs)
    redo = []
    for i, (s, o) in enumerate(zip(segs, outs)):
        if _bad(s, o):
            redo.append(i)
        else:
            res[i] = _decode(o)
    if redo:
        outs2 = _ct2([segs[i].tokens for i in redo], REPEAT_KW)
        for i, o in zip(redo, outs2):
            res[i] = None if _bad(segs[i], o) else _decode(o)
        log.info(f"анти-галлюцинации: повтор {len(redo)}/{len(segs)} сегментов")
    return res


def _gpu_reset():
    try:
        translator.unload_model()
        translator.load_model()
    except Exception as e:
        log.error(f"gpu reset failed: {e}")


def _infer(segs):
    t0 = time.perf_counter()
    try:
        res = _infer_segments(segs)
    except Exception as e:
        log.warning(f"ошибка батча ({len(segs)} сегм.): {e} — повтор по одному")
        _gpu_reset()
        res = []
        for s in segs:
            try:
                res.append(_infer_segments([s])[0])
            except Exception as e2:
                log.error(f"сегмент не переведён: {e2}")
                _gpu_reset()
                res.append(GpuError(str(e2)))
    log.info(f"батч {len(segs)} сегм. за {time.perf_counter() - t0:.2f}с")
    return res


async def batcher():
    loop = asyncio.get_running_loop()
    while True:
        batch = [await queue.get()]
        deadline = loop.time() + BATCH_WINDOW
        while len(batch) < MAX_BATCH:
            t = deadline - loop.time()
            if t <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), t))
            except asyncio.TimeoutError:
                break
        batch.sort(key=lambda s: len(s.tokens))
        try:
            results = await loop.run_in_executor(pool, _infer, batch)
        except Exception as e:
            results = [GpuError(str(e))] * len(batch)
        for s, r in zip(batch, results):
            if s.fut.done():
                continue
            if isinstance(r, Exception):
                s.fut.set_exception(r)
            else:
                s.fut.set_result(r)


def _json(obj, status=200):
    return web.Response(body=orjson.dumps(obj), status=status, content_type="application/json")


async def handle_translate(request: web.Request) -> web.Response:
    if state["stage"] != "ready":
        return _json({"error": state["stage"]}, 503)
    try:
        data = await request.json()
        items = data["items"]
        assert isinstance(items, list) and len(items) <= MAX_ITEMS_PER_REQUEST
    except Exception:
        return _json({"error": "bad request"}, 400)

    loop = asyncio.get_running_loop()
    plans = []
    for it in items:
        text = it.get("text") if isinstance(it, dict) else None
        src = it.get("src") if isinstance(it, dict) else None
        if not isinstance(text, str) or not text.strip() or src not in NLLB_CODES:
            plans.append(None)
            continue
        segs, trunc = make_segments(text, src)
        if not segs:
            plans.append(None)
            continue
        for s in segs:
            s.fut = loop.create_future()
            queue.put_nowait(s)
        plans.append((segs, trunc))

    all_futs = [s.fut for p in plans if p for s in p[0]]
    results = await asyncio.gather(*all_futs, return_exceptions=True)
    if any(isinstance(r, Exception) for r in results):
        return _json({"error": "gpu error"}, 503)

    it_res = iter(results)
    out = []
    for p in plans:
        if p is None:
            out.append(None)
            continue
        segs, trunc = p
        parts = [r for r in (next(it_res) for _ in segs) if r]
        out.append({"text": " ".join(parts), "truncated": trunc} if parts else None)
    return _json({"translations": out})


async def handle_health(request: web.Request) -> web.Response:
    ready = state["stage"] == "ready"
    body = {
        "status": state["stage"],
        "device": state["device"],
        "compute_type": state["compute_type"],
        "vram_used_mb": vram_used_mb(),
        "queue": queue.qsize() if queue else 0,
    }
    return _json(body, 200 if ready else 503)


def _snap(patterns):
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(MODEL_ID, allow_patterns=patterns, local_files_only=True)
    except Exception:
        state["stage"] = "downloading"
        log.info(f"скачивание {MODEL_ID} {patterns}")
        return snapshot_download(MODEL_ID, allow_patterns=patterns)


def _prepare():
    if os.getenv("USE_PROXY", "false").lower() == "true" and os.getenv("PROXY_URL"):
        p = os.environ["PROXY_URL"]
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[k] = p
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"
        log.info(f"HF через прокси {p}")

    tok_patterns = ["*.json", "*.model", "*.txt"]
    done = os.path.join(CT2_DIR, ".done")
    if os.path.exists(done):
        _snap(tok_patterns)
        return

    _snap(tok_patterns + ["*.bin"])
    state["stage"] = "converting"
    log.info("конвертация в CT2 int8 (CPU, ~6 ГБ RAM, несколько минут)")
    if os.path.exists(CT2_DIR):
        shutil.rmtree(CT2_DIR)
    tmp = CT2_DIR + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    subprocess.run(
        ["ct2-transformers-converter", "--model", MODEL_ID,
         "--quantization", "int8", "--output_dir", tmp],
        check=True, env=env,
    )
    open(os.path.join(tmp, ".done"), "w").close()
    os.rename(tmp, CT2_DIR)
    log.info("конвертация завершена")


def _load():
    global translator, tok
    import ctranslate2
    from transformers import AutoTokenizer

    state["stage"] = "loading"
    if ctranslate2.get_cuda_device_count() < 1:
        raise RuntimeError("CTranslate2 не видит CUDA-устройство (проверь драйвер/--gpus/nvidia-container-toolkit)")
    types = ctranslate2.get_supported_compute_types("cuda")
    log.info(f"CT2 {ctranslate2.__version__}, compute types (cuda): {sorted(types)}")
    ct = next((t for t in ("int8", "int8_float32", "float32") if t in types), None)
    if ct is None:
        raise RuntimeError(f"нет подходящего compute_type среди {sorted(types)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    unk = tok.unk_token_id
    missing = [c for c in sorted(NLLB_CODES | {TGT}) if tok.convert_tokens_to_ids(c) in (None, unk)]
    if missing:
        raise RuntimeError(f"коды отсутствуют в токенизаторе NLLB: {missing}")

    translator = ctranslate2.Translator(
        CT2_DIR, device="cuda", compute_type=ct, inter_threads=1, intra_threads=2
    )
    state["compute_type"] = ct
    log.info(f"модель загружена, compute_type={ct}")


def _warmup():
    samples = [
        ("Привет, как дела? Сегодня отличная погода.", "rus_Cyrl"),
        ("Guten Morgen, wie geht es Ihnen heute?", "deu_Latn"),
        ("今天天气很好，我们一起去公园散步吧。", "zho_Hans"),
    ]
    segs = []
    for text, src in samples:
        s, _ = make_segments(text, src)
        segs.extend(s)
    t0 = time.perf_counter()
    res = _infer(segs)
    for r in res:
        if isinstance(r, Exception) or not r:
            raise RuntimeError(f"прогрев не удался: {res}")
    log.info(f"прогрев {time.perf_counter() - t0:.2f}с: {res}")
    log.info(f"VRAM (всего на устройстве) после прогрева: {vram_used_mb()} МБ")


async def init():
    loop = asyncio.get_running_loop()
    try:
        state["stage"] = "downloading"
        await loop.run_in_executor(None, _prepare)
        await loop.run_in_executor(pool, _load)
        await loop.run_in_executor(pool, _warmup)
        state["stage"] = "ready"
        log.info("✅ translator ready")
    except Exception:
        log.exception("❌ инициализация translator провалилась")
        os._exit(1)


async def on_startup(app: web.Application):
    global queue
    queue = asyncio.Queue()
    asyncio.create_task(batcher())
    asyncio.create_task(init())


app = web.Application(client_max_size=64 * 1024 * 1024)
app.router.add_post("/translate", handle_translate)
app.router.add_get("/health", handle_health)
app.on_startup.append(on_startup)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=PORT, print=None, access_log=None)