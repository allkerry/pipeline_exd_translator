# Exorde Pipeline — Debian 13 + NVIDIA P106-100 (6 ГБ) + перевод NLLB

Схема:
```
collector:9000 → upipe:5981 → bpipe:7995 → transactioneer:8002 → Exorde
                    │                              │
                    ▼ не-en                        ▼
            translator:8003                  vpn (Xray/REALITY)
```
Порядок старта: `vpn → translator → transactioneer → bpipe → upipe → collector`. Одна GPU делится между `translator` и `bpipe`.

- **collector** — фильтры длины/дубликатов/возраста, обрезка пейлоада.
- **upipe** — детект языка, перевод не-en через `translator`, keywords, **отброс item с превышением токен-лимита**.
- **translator** — NLLB-200-distilled-1.3B, CTranslate2 int8, GPU. Любой из 54 языков langdetect → английский. Английские тексты в него не ходят.
- **bpipe** — эмбеддинги, sentiment, emotion, zero-shot и т.д. на GPU.
- **transactioneer** — грузит батчи в Exorde Upload API **через VPN**.
- **vpn** — Xray-core (VLESS + REALITY + Vision), SOCKS5 :1080 и HTTP :1081 внутри docker-сети.

Лицензия NLLB — CC-BY-NC-4.0 (только некоммерческое использование).

---

## 1. Подготовка голого Debian 13

Под root (`su -`):
```bash
apt update && apt install -y sudo
usermod -aG sudo ТВОЙ_ЛОГИН
exit
```
Перелогинься, дальше под обычным пользователем.

Включить non-free (нужен для драйвера NVIDIA):
```bash
sudo sed -i 's/^Components: main.*/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources
# если файла debian.sources нет (старый формат):
# sudo sed -i 's/ main$/ main contrib non-free non-free-firmware/' /etc/apt/sources.list
sudo apt update
```

Драйвер (ветка Debian 13 = 550.x, Pascal поддерживается; драйверы >580 Pascal не поддерживают — не ставь их с сайта NVIDIA):
```bash
sudo apt install -y linux-headers-amd64 nvidia-driver firmware-misc-nonfree
sudo reboot
```
Если включён Secure Boot — отключи в BIOS, иначе модуль драйвера не загрузится.

После ребута `nvidia-smi` должен показать **P106-100**. Карта без видеовыхода: монитор на встроенную графику или работа по SSH, ничего настраивать не нужно.

```bash
sudo apt install -y ca-certificates curl gnupg git

# ─── Docker + Compose plugin ───────────────────────────────────
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker

# ─── NVIDIA Container Toolkit (чтобы Docker видел GPU) ─────────
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Проверка (должна показать P106-100):
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 2. Настройка

Положи проект в `~/exorde-pipeline` (папки `collector/ upipe/ bpipe/ transactioneer/ vpn/ translator/` и файлы `docker-compose.yml run.sh`):
```bash
cd ~/exorde-pipeline
chmod +x run.sh
cp .env.example .env
nano .env        # MAIN_ADDRESS и VLESS_URL — обязательно
```
Ключевое в `.env`:
- `LANG_FILTER=` — пусто: фильтр collector выключен (иначе не-en режется до перевода).
- `BPIPE_TORCH_DTYPE=float32` — на Pascal fp16 ~в 64 раза медленнее fp32.

---

## 3. Запуск

```bash
./run.sh up -d --build
docker compose logs -f translator
docker compose ps
```
Первый старт 15–40 мин: сборка образов, скачивание NLLB (~5.5 ГБ), конвертация в int8 (CPU, ~6 ГБ RAM), затем модели bpipe. Стадии translator: `downloading → converting → loading → ready`. Повторный старт ≤3 мин (кеш в `./models_cache`).
Должно быть 6 сервисов (`vpn`, `translator`, `transactioneer`, `bpipe`, `upipe`, `collector`) в статусе `healthy`.

---

## 4. Проверка

### 4.1. VPN
```bash
docker compose logs vpn | tail -20          # "✅ Конфиг Xray собран: ... pq=да"
curl -x socks5h://127.0.0.1:1080 https://api.ipify.org   # IP VPN-сервера, не твой
```
Если не поднимается — в `.env` `VPN_DISABLE_PQV=true` и `docker compose up -d --build vpn`.

### 4.2. /health
```bash
curl -sf http://localhost:9000/health   # collector
docker compose exec upipe curl -sf http://localhost:5981/health
docker compose exec bpipe curl -sf http://localhost:7995/health
docker compose exec transactioneer curl -sf http://localhost:8002/health
docker compose exec translator curl -s http://localhost:8003/health
```
Translator: `"status":"ready"`, `"device":"cuda"`, `"compute_type":"int8"`.

### 4.3. Перевод
```bash
for t in "Это тестовое сообщение на русском языке о погоде сегодня." \
         "Das ist eine Testnachricht über das heutige Wetter in Berlin." \
         "今天的天气非常好，我们打算去公园散步，然后喝咖啡。" \
         "Bu hava durumu hakkında bir test mesajıdır, bugün hava çok güzel."; do
  curl -s -X POST http://localhost:9000/store_item -H "Content-Type: application/json" \
    -d "{\"content\":\"$t\",\"external_id\":\"tr-$RANDOM\",\"created_at\":\"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\",\"domain\":\"twitter.com\",\"url\":\"https://twitter.com/t/status/1\"}"; echo
done
docker compose logs --since 2m translator upipe | tail -30
```
Ожидается: в логах translator `батч N сегм. за …с`, затем батч в bpipe и `✅ Загружено` в transactioneer. Английский текст translator не вызывает.

### 4.4. Английский текст
```bash
curl -X POST http://localhost:9000/store_item -H "Content-Type: application/json" \
  -d '{"content":"This is a completely normal test tweet about the weather being sunny today in California.","external_id":"test-normal-001","created_at":"'"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"'","domain":"twitter.com","url":"https://twitter.com/test/status/1"}'
```
Ответ `{"message": "OK"}`.

### 4.5. Короткий текст
Текст короче `MIN_TEXT_LEN` → `{"message": "skipped_short"}`, до GPU не доходит.

### 4.6. Очень длинный английский текст
Item сверх `MAX_MODEL_TOKENS` (400) **отбрасывается целиком** в upipe (`Токен-лимит превышен`). Переведённый текст, наоборот, обрезается по предложениям до лимита и не отбрасывается.

### 4.7. Мусор / пустой текст
`"!!!!!!!!!! ################ 1234567890"` → в логах upipe `No content to work with`, сервисы не падают.

### 4.8. Отказоустойчивость
`docker compose stop translator` → upipe не падает, растёт счётчик `errors` в его `/health`; `docker compose start translator` → всё восстанавливается.

---

## 5. VRAM

```bash
watch -n 1 nvidia-smi
```
Цель: суммарно ≤5.4 ГБ из 6 под нагрузкой. **Замеров пока нет — внеси сюда свои цифры.**
Оценка: translator ~2.0–2.5 ГБ + bpipe в `float32` ~2.5–3 ГБ → ~5–5.5 ГБ, впритык.
Лестница отката (после каждой ступени — перезамер, `./run.sh up -d`):
1. `.env`: `FIXED_BATCH_SIZE=32`, `TAG_BATCH_SIZE=8`, `TRANSLATOR_MAX_BATCH=4`.
2. `.env`: `MAX_TRANSLATE_SRC_TOKENS=400`, `TRANSLATOR_BEAM=2`.
3. `.env`: `NLLB_MODEL=facebook/nllb-200-distilled-600M` и `NLLB_CT2_DIR=nllb-600m-distilled-ct2-int8`.

Вторая реплика bpipe (`BPIPE_REPLICAS=2`) — только если измеренный пик ≤5.4 ГБ.

---

## 6. Повседневные команды

```bash
docker compose logs -f [сервис]
docker compose restart bpipe
docker compose down
./run.sh up -d --build         # пересобрать после правок кода
docker compose down -v         # остановить + стереть тома
rm -rf models_cache            # стереть кеш моделей (переконвертация NLLB заново)
```

## 7. Если что-то не так

- **translator долго не healthy** — норма при первом старте; смотри `docker compose logs -f translator`.
- **CUDA out of memory** — лестница отката из раздела 5.
- **transactioneer не отправляет** — проверь VPN (4.1) и `MAIN_ADDRESS`.
- **HuggingFace не скачивается** — `USE_PROXY_TRANSLATOR=true`, `USE_PROXY_BPIPE=true`, `USE_PROXY_UPIPE=true` в `.env`, затем `./run.sh up -d --build`.
- **`Illegal instruction` в логах** — CPU без AVX2 (i7-3770K): сообщи, какой контейнер.

## 8. Нюансы

- `MIN_TEXT_LEN=20` символов для CJK слишком много (20 иероглифов — уже длинное предложение): при необходимости снизь в `.env`.
- `langdetect` путает близкие языки (hi/mr/ne, hr/sr, id/ms) и короткие тексты; порог `LANG_CONFIDENCE_THRESHOLD=0.90`. Не-en с уверенностью ниже порога или вне таблицы 54 языков отбрасывается (`filtered_lang`).
- Для zh/ja/ko/th анти-галлюцинационный порог длины вывода — 6× токенов входа (для остальных 3×).
- `test_pipeline.sh` обновлён: тест №9 проверяет вызов translator, лимит GPU — `GPU_LIMIT_MIB` (по умолчанию 6144).
- Расхождение README ↔ `upipe/process.py` для английского: item сверх лимита отбрасывается, а не обрезается. Не менялось.