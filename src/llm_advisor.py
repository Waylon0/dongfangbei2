"""大模型顾问 — DeepSeek API 封装，提供参数推荐和地质解读。

API Key 通过环境变量 DEEPSEEK_API_KEY 传入。
"""

import os
import json
import requests

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_CHAT = f"{DEEPSEEK_BASE}/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"  # DeepSeek-V4
TIMEOUT = 30
_HARDCODED_KEY = "sk-6ef2e89936d74cb490fe143e4f4cecfd"


def _api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key
    return _HARDCODED_KEY


def _call(messages: list[dict], stream: bool = False) -> str | None:
    """调用 DeepSeek API，失败返回 None。"""
    key = _api_key()
    if not key:
        return None
    try:
        resp = requests.post(
            DEEPSEEK_CHAT,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEFAULT_MODEL,
                "messages": messages,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        return None
    except Exception:
        return None


def recommend_params(data_stats: dict, config_dict: dict) -> str | None:
    """根据数据统计特征和当前参数，返回 LLM 参数调整建议。"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是地球物理断层解释专家。用户正在进行基于平面断层属性的断层多边形自动追踪。"
                "请根据用户提供的数据统计特征和当前算法参数，给出3-5条具体的参数调整建议。"
                "每条建议应包含：参数名、建议值、理由（1-2句）。"
                "语气简洁专业，不要客套话。"
                "如果参数已经合理，也要说明当前设置是合适的。"
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
                f"- 噪声估计(局部方差中位数): {data_stats.get('noise_est', 0):.4f}\n"
                f"\n## 当前参数\n"
                f"```json\n{json.dumps(config_dict, ensure_ascii=False, indent=2)}\n```\n"
                f"\n请给出参数调整建议。"
            ),
        },
    ]
    return _call(messages)


def interpret_results(result_stats: dict) -> str | None:
    """根据检测结果统计，返回 LLM 地质解读。"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是构造地质学家，擅长解读断层检测结果。"
                "请根据用户提供的断层检测统计信息，从地质角度进行2-3段解读，包括："
                "1) 断层分布特征（数量、规模、走向趋势）；"
                "2) 与地质认识的吻合度评估；"
                "3) 可能的改进方向。"
                "语气简洁专业，不要客套话。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## 检测结果统计\n"
                f"- 断层数量: {result_stats.get('count', 0)}\n"
                f"- 总面积: {result_stats.get('total_area', 0):.1f} 像素²\n"
                f"- 总长度(估算): {result_stats.get('total_length', 0):.1f} 像素\n"
                f"- 最小面积: {result_stats.get('min_area', 0):.1f} 像素²\n"
                f"- 最大面积: {result_stats.get('max_area', 0):.1f} 像素²\n"
                f"- 面积中位数: {result_stats.get('median_area', 0):.1f} 像素²\n"
                f"- 处理耗时: {result_stats.get('elapsed', 0):.1f}s\n"
                f"- 算法模式: {result_stats.get('mode', 'unknown')}\n"
                f"\n请给出地质解读。"
            ),
        },
    ]
    return _call(messages)


def chat(prompt: str, context: dict | None = None) -> str | None:
    """自由问答，可附加上下文（数据统计 + 结果统计）。"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是DeepSeek-V4大模型驱动的AI助手，能够自由回答任何问题。"
                "当用户讨论断层检测相关话题时，你可以结合以下背景知识提供专业建议：\n"
                "- 地震属性分析与断层检测原理\n"
                "- 图像处理算法（Otsu二值化、高斯滤波、形态学、骨架化、Douglas-Peucker简化等）\n"
                "- 构造地质学基础（断层分类、走向分析、应力场解释）\n"
                "- 软件参数调优经验\n\n"
                "回答要求：\n"
                "1. 自由回答任何问题，不限制话题范围\n"
                "2. 如果问题与断层检测相关，优先结合专业知识和当前上下文回答\n"
                "3. 始终使用中文回答，语气专业但平易近人\n"
                "4. 回答简洁有力，一般不超过5段"
            ),
        },
    ]
    if context:
        ctx_text = "## 当前上下文\n"
        if "data_stats" in context:
            ctx_text += f"数据: {json.dumps(context['data_stats'], ensure_ascii=False)}\n"
        if "result_stats" in context:
            ctx_text += f"结果: {json.dumps(context['result_stats'], ensure_ascii=False)}\n"
        if "config" in context:
            ctx_text += f"参数: {json.dumps(context['config'], ensure_ascii=False)}\n"
        messages.append({"role": "user", "content": ctx_text})
        messages.append({"role": "assistant", "content": "已了解当前数据和参数情况，请提问。"})
    messages.append({"role": "user", "content": prompt})
    return _call(messages)
