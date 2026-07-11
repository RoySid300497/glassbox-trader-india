"""calling free-tier llm providers through one self-updating interface"""

import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 600
TIMEOUT = 45

# per-provider config: where to chat, where to discover models, small
# ranked hints (encoding free-tier quota shape and model-family
# decorrelation that catalogs cannot express), and a last-resort static
# fallback used only if discovery itself is down
PROVIDER_CONFIG = {
    "gemini": {
        "key_env": "GEMINI_API_KEY", "model_env": "GEMINI_MODEL",
        "models_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "hints": ["gemini-2.5-flash", "gemini-2.0-flash"],
        "fallback": "gemini-2.5-flash",
    },
    "groq": {
        "key_env": "GROQ_API_KEY", "model_env": "GROQ_MODEL",
        "chat_url": "https://api.groq.com/openai/v1/chat/completions",
        "models_url": "https://api.groq.com/openai/v1/models",
        "hints": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "fallback": "llama-3.3-70b-versatile",
    },
    "mistral": {
        "key_env": "MISTRAL_API_KEY", "model_env": "MISTRAL_MODEL",
        "chat_url": "https://api.mistral.ai/v1/chat/completions",
        "models_url": "https://api.mistral.ai/v1/models",
        "hints": ["mistral-small-latest", "mistral-medium-latest"],
        "fallback": "mistral-small-latest",
    },
    "cerebras": {
        "key_env": "CEREBRAS_API_KEY", "model_env": "CEREBRAS_MODEL",
        "chat_url": "https://api.cerebras.ai/v1/chat/completions",
        "models_url": "https://api.cerebras.ai/v1/models",
        "hints": ["zai-glm-4.7", "qwen-3-235b-a22b-instruct-2507",
                  "qwen-3-32b"],
        "fallback": "zai-glm-4.7",
    },
    "sambanova": {
        "key_env": "SAMBANOVA_API_KEY", "model_env": "SAMBANOVA_MODEL",
        "chat_url": "https://api.sambanova.ai/v1/chat/completions",
        "models_url": "https://api.sambanova.ai/v1/models",
        "hints": ["DeepSeek-V3.2", "DeepSeek-V3.1"],
        "fallback": "DeepSeek-V3.2",
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY", "model_env": "OPENROUTER_MODEL",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
        "models_url": "https://openrouter.ai/api/v1/models",
        "require": ":free",
        # openrouter's own free-pool router load-balances across whatever
        # free capacity is live, dodging single-model upstream congestion
        "hints": ["qwen/qwen3-235b-a22b:free",
                  "meta-llama/llama-4-maverick:free",
                  "meta-llama/llama-4-scout:free"],
        "fallback": "openrouter/auto",
        "prefer_router": "openrouter/auto",
        "min_size": 30,
    },
    "nvidia": {
        "key_env": "NVIDIA_API_KEY", "model_env": "NVIDIA_MODEL",
        "chat_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "models_url": "https://integrate.api.nvidia.com/v1/models",
        "hints": ["nvidia/nemotron-3-ultra-550b-a55b",
                  "nvidia/llama-3.3-nemotron-super-49b-v1.5",
                  "nvidia/llama-3.1-nemotron-70b-instruct"],
        "fallback": "nvidia/nemotron-3-ultra-550b-a55b",
        # nemotron is a reasoning model: thinking must stay off for judge
        # seats or the json lands in the reasoning channel past our budget
        "extra_json": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "github_models": {
        "key_env": "GH_MODELS_TOKEN", "model_env": "GH_MODELS_MODEL",
        "chat_url": "https://models.github.ai/inference/chat/completions",
        "models_url": "https://models.github.ai/catalog/models",
        "hints": ["openai/gpt-4.1-mini", "openai/gpt-4o-mini"],
        "fallback": "openai/gpt-4.1-mini",
    },
}

BENCH_PROVIDERS = ["cerebras", "sambanova", "github_models",
                   "openrouter", "nvidia"]

# models that are not general chat judges get filtered before ranking
_EXCLUDE = ("embed", "whisper", "tts", "audio", "image", "vision", "ocr",
            "guard", "moderation", "rerank", "code", "coder", "transcrib",
            "realtime", "live", "safety", "classifier")


def _model_size(model_id):
    # reading the parameter count in billions from a model id, if present
    import re
    mid = model_id.lower()
    best = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([bm])", mid):
        v = float(num) * (0.001 if unit == "m" else 1)
        best = max(best, v)
    return best


def _rank_score(model_id):
    # scoring a model id: newer generation and bigger size rank higher
    mid = model_id.lower()
    if any(x in mid for x in _EXCLUDE):
        return -1.0
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", mid)]
    generation = max([n for n in nums if n < 10], default=0.0)
    size = max([n for n in nums if 7 <= n <= 2000], default=0.0)
    score = generation * 10 + min(size, 500) / 100
    if "latest" in mid:
        score += 2
    if "instruct" in mid or "versatile" in mid or "chat" in mid:
        score += 1
    if "distill" in mid or "mini" in mid or "nano" in mid or "lite" in mid:
        score -= 1
    if "preview" in mid or "exp" in mid:
        score -= 2
    return score


def _catalog_ids(provider):
    # asking the provider which model ids this key can actually use
    cfg = PROVIDER_CONFIG[provider]
    key = os.environ.get(cfg["key_env"])
    if provider == "gemini":
        r = requests.get(f"{cfg['models_url']}?key={key}", timeout=20)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:120]}")
        out = []
        for m in r.json().get("models", []):
            if "generateContent" in (m.get("supportedGenerationMethods")
                                     or []):
                out.append(m.get("name", "").removeprefix("models/"))
        return out
    r = requests.get(cfg["models_url"],
                     headers={"Authorization": f"Bearer {key}"}, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text[:120]}")
    body = r.json()
    items = body.get("data", body) if isinstance(body, dict) else body
    return [m.get("id") for m in items if isinstance(m, dict) and m.get("id")]


_resolved_models = {}


def resolve_model(provider):
    # choosing a model once per run: a catalog-confirmed env pin, then the
    # first hint the live catalog offers, then the best-ranked catalog
    # entry, then the static fallback — a stale pin degrades, never kills
    if provider in _resolved_models:
        return _resolved_models[provider]
    cfg = PROVIDER_CONFIG[provider]
    available = None
    try:
        available = _catalog_ids(provider)
    except Exception as e:
        print(f"  [llm] {provider} model discovery failed ({e})")

    pick = None
    router = cfg.get("prefer_router")
    if router and not os.environ.get(cfg["model_env"]):
        # trying the provider's own load-balancing router first
        pick_router = router
    else:
        pick_router = None
    pinned = os.environ.get(cfg["model_env"])
    if pinned:
        confirmed = available is None or pinned in available or any(
            pinned in a or a in pinned for a in available)
        if confirmed:
            pick = pinned
        else:
            print(f"  [llm] {provider}: pinned model '{pinned}' not in "
                  f"live catalog — ignoring the pin")
    if available and cfg.get("require"):
        available = [a for a in available if cfg["require"] in a]
    if not pick and available:
        pick = next((h for h in cfg["hints"] if h in available), None)
        if not pick:
            floor = cfg.get("min_size", 0)
            ranked = [m for m in sorted(available, key=_rank_score,
                                        reverse=True)
                      if _rank_score(m) > 0 and _model_size(m) >= floor]
            if ranked:
                pick = ranked[0]
                print(f"  [llm] {provider}: no hint available — "
                      f"auto-ranked {pick} from {len(available)} models")
    if not pick and pick_router:
        pick = pick_router
        print(f"  [llm] {provider}: using load-balancing router {pick}")
    if not pick:
        pick = cfg["fallback"]
        print(f"  [llm] {provider}: using static fallback {pick}")
    _resolved_models[provider] = pick
    return pick


def _openai_style(provider, prompt):
    # calling any openai-compatible endpoint with shared plumbing
    cfg = PROVIDER_CONFIG[provider]
    payload = {"model": resolve_model(provider),
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": MAX_TOKENS, "temperature": 0.4}
    payload.update(cfg.get("extra_json", {}))
    r = requests.post(cfg["chat_url"],
                      headers={"Authorization":
                               f"Bearer {os.environ.get(cfg['key_env'])}"},
                      json=payload,
                      timeout=TIMEOUT)
    if r.status_code >= 400:
        # surfacing the body so a 4xx explains itself in the log
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]


def ask_gemini(prompt):
    # requesting one completion from gemini on its resolved model
    key = os.environ.get("GEMINI_API_KEY")
    model = resolve_model("gemini")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    r = requests.post(url,
                      json={"contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "maxOutputTokens": MAX_TOKENS,
                                "temperature": 0.4,
                                "thinkingConfig": {"thinkingBudget": 0}}},
                      timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


PROVIDERS = {
    "gemini": ask_gemini,
    "openrouter": lambda p: _openai_style("openrouter", p),
    "nvidia": lambda p: _openai_style("nvidia", p),
    "groq": lambda p: _openai_style("groq", p),
    "mistral": lambda p: _openai_style("mistral", p),
    "cerebras": lambda p: _openai_style("cerebras", p),
    "sambanova": lambda p: _openai_style("sambanova", p),
    "github_models": lambda p: _openai_style("github_models", p),
}

# bench providers only join rotation when their key is present
BENCH = [p for p in BENCH_PROVIDERS
         if os.environ.get(PROVIDER_CONFIG[p]["key_env"])]

# explicit substitution priority: deepest free quotas first, gemini last;
# override with a FALLBACK_ORDER repo variable (comma-separated)
DEFAULT_FALLBACK_ORDER = ["cerebras", "groq", "mistral", "nvidia",
                          "github_models", "openrouter", "sambanova",
                          "gemini"]


def _fallback_order():
    raw = os.environ.get("FALLBACK_ORDER")
    order = ([p.strip() for p in raw.split(",") if p.strip()]
             if raw else DEFAULT_FALLBACK_ORDER)
    known = [p for p in order if p in PROVIDER_CONFIG]
    keyed = [p for p in known
             if os.environ.get(PROVIDER_CONFIG[p]["key_env"])]
    # anything keyed but unlisted still belongs at the back of the line
    keyed += [p for p in PROVIDER_CONFIG
              if p not in keyed
              and os.environ.get(PROVIDER_CONFIG[p]["key_env"])]
    return keyed


FALLBACK_ORDER = _fallback_order()


_consecutive_failures = {}
_circuit_opened_at = {}
CIRCUIT_THRESHOLD = 3        # opening the circuit after this many strikeouts
CIRCUIT_COOLDOWN = 120       # allowing a single probe call after this long
QUOTA_COOLDOWN = 3600        # probing only hourly once a daily quota is dead
_quota_exhausted = {}


def _quota_dead(status, message):
    # recognising a spent daily quota, which no backoff can fix
    if status != 429:
        return False
    m = message.lower()
    return ("current quota" in m or "plan and billing" in m
            or "perday" in m or "daily" in m)


def ask(provider, prompt):
    # routing one prompt with 429-aware backoff and a half-open circuit
    # breaker that probes for recovery instead of staying dark all run
    import time
    probing = False
    if _consecutive_failures.get(provider, 0) >= CIRCUIT_THRESHOLD:
        cooldown = (QUOTA_COOLDOWN if _quota_exhausted.get(provider)
                    else CIRCUIT_COOLDOWN)
        if time.time() - _circuit_opened_at.get(provider, 0) < cooldown:
            print(f"  [llm] {provider} circuit open — skipping")
            return None
        probing = True
        print(f"  [llm] {provider} circuit half-open — probing for recovery")

    attempts = (1,) if probing else (1, 2, 3)
    for attempt in attempts:
        try:
            reply = PROVIDERS[provider](prompt)
            if _consecutive_failures.get(provider, 0) >= CIRCUIT_THRESHOLD:
                print(f"  [llm] {provider} recovered — circuit closed")
            _consecutive_failures[provider] = 0
            _quota_exhausted.pop(provider, None)
            return reply
        except Exception as e:
            status = getattr(getattr(e, "response", None),
                             "status_code", None)
            if status is None and str(e)[:3] == "429":
                status = 429
            print(f"  [llm] {provider} attempt {attempt} failed: {e}")
            if _quota_dead(status, str(e)):
                # a spent daily quota cannot be waited out — benching now
                print(f"  [llm] {provider} daily quota exhausted — "
                      f"benching for this run")
                _quota_exhausted[provider] = True
                _consecutive_failures[provider] = CIRCUIT_THRESHOLD
                _circuit_opened_at[provider] = time.time()
                return None
            if attempt < attempts[-1]:
                # waiting out a rate-limit window instead of burning strikes
                time.sleep(20 if status == 429 else 4)
    _consecutive_failures[provider] = \
        _consecutive_failures.get(provider, 0) + 1
    _circuit_opened_at[provider] = time.time()
    return None


def parse_json_reply(text, required_keys):
    # extracting and validating strict json from an llm reply
    if not text:
        return None
    try:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        obj = json.loads(text[start:end + 1])
        if not all(k in obj for k in required_keys):
            return None
        return obj
    except (json.JSONDecodeError, TypeError):
        return None
