"""Central configuration: filesystem layout, API channels, prompts, registries.

This is the single place that knows where data lives and how to reach the judge
API. Everything is driven by environment variables (optionally a ``.env`` file
at the bundle root), so the tool is portable and contains no secrets.

Environment variables
---------------------
API endpoint (OpenAI-compatible; required to run the judge):
    MODEL_REQUEST_URL   base URL, e.g. ``https://az.gptplus5.com/v1``
    API_KEY             API key for that endpoint
    API_KEYS            optional, comma-separated keys; each becomes its own
                        concurrent channel on MODEL_REQUEST_URL (overrides API_KEY)
Data location:
    OOD_DATA_ROOT       datasets + outputs root (default: ``<bundle>/data``)
Concurrency:
    MAX_CONCURRENCY_PER_KEY   per-key in-flight cap (default 6)
    GLOBAL_MAX_CONCURRENCY    machine-wide in-flight cap (default 16)
"""
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))     # .../benchmark_analysis
BUNDLE_DIR = os.path.dirname(PKG_DIR)                     # bundle root
PROMPTS_DIR = os.path.join(PKG_DIR, "prompts")
DEFAULT_PROMPT = "multi_v4.txt"


def _load_dotenv(path):
    """Load ``KEY=VALUE`` lines from a .env file without overriding real env vars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(BUNDLE_DIR, ".env"))

DATA_ROOT = os.path.abspath(
    os.environ.get("OOD_DATA_ROOT", os.path.join(BUNDLE_DIR, "data")))
ANALYSIS_DIR = os.path.join(DATA_ROOT, "analysis")
GLOBAL_MAX_CONCURRENCY = int(os.environ.get("GLOBAL_MAX_CONCURRENCY", "16"))

# Judge models we recognise when splitting "<agent>_<model>" result filenames.
# Unknown models still work (treated as a single agent-less model).
KNOWN_MODELS = [
    "gemini-3.1-pro-preview", "gemini-3-flash-preview", "claude-sonnet-4-6",
    "claude-opus-4-8", "gpt-5-mini", "gpt-5.5", "qwen3.5-397b-a17b",
    "qwen3.5-35b-a3b", "qwen3-vl-235b-a22b-instruct", "qwen3-vl-30b-a3b-instruct",
]
# Rollout agents excluded from the pooled leaderboard (A/B control arms).
EXCLUDE_AGENTS = {"ossymphony-full"}


def data_path(*parts):
    """A path under the dataset root (raw inputs, analysis outputs, ...)."""
    return os.path.join(DATA_ROOT, *parts)


def platform_dir(platform):
    """analysis/<platform> output root (judge_ready / images / results / ...)."""
    return os.path.join(ANALYSIS_DIR, platform)


def prompt_path(name=DEFAULT_PROMPT):
    return os.path.join(PROMPTS_DIR, name)


def load_system_prompt(name=DEFAULT_PROMPT):
    """Read a judge system prompt verbatim from the bundled prompts/ folder."""
    with open(prompt_path(name), encoding="utf-8") as f:
        return f.read()


def get_channels():
    """Build the list of API channels from the environment.

    Returns a list of dicts ``{name, base_url, api_key, max_concurrency}``.
    Raises if no endpoint/key is configured.
    """
    base = os.environ.get("MODEL_REQUEST_URL", "").strip()
    keys = [k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()]
    if not keys:
        single = os.environ.get("API_KEY", "").strip()
        keys = [single] if single else []
    if not base or not keys:
        raise RuntimeError(
            "No API channel configured. Set MODEL_REQUEST_URL and API_KEY "
            "(or API_KEYS) in the environment or in a .env file at the bundle root.")
    per_key = int(os.environ.get("MAX_CONCURRENCY_PER_KEY", "6"))
    return [{"name": f"ch{i + 1}", "base_url": base, "api_key": k,
             "max_concurrency": per_key} for i, k in enumerate(keys)]
