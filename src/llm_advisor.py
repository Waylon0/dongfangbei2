"""AI 助手 — DeepSeek API 流式聊天 + Bing 联网搜索

API Key: 环境变量 DEEPSEEK_API_KEY，未设置时用内置 key。
"""

import os
import json
import re
import requests
from urllib.parse import quote

# ── 配置 ──────────────────────────────────────────────

DEEPSEEK_CHAT = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
TIMEOUT = 30
_HARDCODED_KEY = ""  # 请设置环境变量 DEEPSEEK_API_KEY，不要在此处填写 key

SYSTEM_PROMPT = (
    "你是 DeepSeek AI 助手，知识渊博、乐于助人。"
    "回答简洁准确，适当使用 Markdown 格式让排版清晰。"
)


def _api_key():
    return os.environ.get("DEEPSEEK_API_KEY", "") or _HARDCODED_KEY


# ── 流式聊天 ──────────────────────────────────────────

def chat_stream(messages: list[dict], max_tokens: int = 1024):
    """流式调用 DeepSeek，逐块 yield 文本。yield None 表示失败。"""
    key = _api_key()
    if not key:
        yield None
        return
    try:
        resp = requests.post(
            DEEPSEEK_CHAT,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={
                "model": DEFAULT_MODEL,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": max_tokens,
            },
            timeout=TIMEOUT, stream=True,
        )
        if resp.status_code != 200:
            yield None
            return
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                delta = json.loads(data_str)["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    except Exception:
        yield None


# ── 联网搜索（Bing 抓取） ─────────────────────────────

def web_search(query: str, num_results: int = 5) -> str:
    """从 Bing 抓取搜索结果摘要。失败返回空字符串。"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        r = requests.get(
            f"https://cn.bing.com/search?q={quote(query)}",
            headers=headers, timeout=10,
        )
        if r.status_code != 200:
            return ""
        snippets = re.findall(r'<p[^>]*>(.*?)</p>', r.text, re.DOTALL)
        results = []
        for s in snippets:
            clean = re.sub(r'<[^>]+>', '', s).strip()
            clean = re.sub(r'\s+', ' ', clean)
            # 清理 HTML 实体
            clean = clean.replace("&ensp;", " ").replace("&emsp;", " ").replace("&nbsp;", " ")
            clean = re.sub(r'&#\d+;', '', clean)
            clean = clean.strip()
            if len(clean) < 30 or clean in results:
                continue
            if any(kw in clean for kw in ['cookie', 'javascript', 'function(',
                                            'var ', 'href=', 'sj_log']):
                continue
            results.append(clean)
            if len(results) >= num_results:
                break
        if not results:
            return ""
        return "以下是最新的网络搜索结果，请基于这些信息回答用户问题：\n" + \
               "\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
    except Exception:
        return ""


# ── 非流式调用（快捷功能用） ──────────────────────────

def _call(messages: list[dict], max_tokens: int = 512) -> str | None:
    """非流式调用，返回完整回复或 None。"""
    key = _api_key()
    if not key:
        return None
    try:
        resp = requests.post(
            DEEPSEEK_CHAT,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={
                "model": DEFAULT_MODEL,
                "messages": messages,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": max_tokens,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return None
    except Exception:
        return None


# ── 快捷功能：参数推荐 ─────────────────────────────────

def recommend_params(data_stats: dict, config_dict: dict) -> str | None:
    """根据数据统计和当前参数，返回参数调整建议。"""
    return _call([
        {
            "role": "system",
            "content": (
                "你是地球物理断层解释专家。用户在进行基于平面断层属性的断层多边形自动追踪。"
                "请根据数据统计特征和当前算法参数，给出3-5条具体的参数调整建议。"
                "每条建议包含：参数名、建议值、理由。语气简洁专业。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## 数据统计\n"
                f"- 网格尺寸: {data_stats.get('rows', '?')} × {data_stats.get('cols', '?')}\n"
                f"- 值范围: [{data_stats.get('vmin', 0):.4f}, {data_stats.get('vmax', 1):.4f}]\n"
                f"- 均值: {data_stats.get('mean', 0):.4f}\n"
                f"- 标准差: {data_stats.get('std', 0):.4f}\n"
                f"- 非零占比: {data_stats.get('nonzero_ratio', 0) * 100:.1f}%\n"
                f"- 噪声估计: {data_stats.get('noise_est', 0):.4f}\n"
                f"\n## 当前参数\n"
                f"```json\n{json.dumps(config_dict, ensure_ascii=False, indent=2)}\n```"
            ),
        },
    ])


# ── 快捷功能：地质解读 ─────────────────────────────────

def interpret_results(result_stats: dict) -> str | None:
    """根据检测结果统计，返回地质解读。"""
    return _call([
        {
            "role": "system",
            "content": (
                "你是构造地质学家，擅长解读断层检测结果。"
                "请从地质角度进行2-3段解读：分布特征、地质吻合度、改进方向。"
                "语气简洁专业。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## 检测结果\n"
                f"- 断层数量: {result_stats.get('count', 0)}\n"
                f"- 总面积: {result_stats.get('total_area', 0):.1f} 像素²\n"
                f"- 总长度(估算): {result_stats.get('total_length', 0):.1f} 像素\n"
                f"- 面积范围: [{result_stats.get('min_area', 0):.1f}, "
                f"{result_stats.get('max_area', 0):.1f}]\n"
                f"- 面积中位数: {result_stats.get('median_area', 0):.1f}\n"
                f"- 处理耗时: {result_stats.get('elapsed', 0):.1f}s\n"
                f"- 算法模式: {result_stats.get('mode', 'unknown')}"
            ),
        },
    ])
