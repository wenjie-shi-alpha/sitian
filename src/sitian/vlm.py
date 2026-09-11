"""按需 VLM 客户端（OpenAI 兼容 vision 接口，纯标准库）。

用法：单卡上临时唤起一个不大的 VLM（推荐 Qwen2.5-VL-7B-Instruct，
训练间隙可与 8B 策略共卡；离线批处理可换 32B），用完即停：

    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001
    export FH_VLM_BASE_URL=http://127.0.0.1:8001/v1 FH_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

环境里 describe_image 工具在未配置 VLM 时返回 available=false，不阻塞 episode。
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from typing import Optional


class VLMClient:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 120.0):
        self.base_url = (base_url or os.environ.get("FH_VLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("FH_VLM_MODEL", "")
        self.api_key = api_key or os.environ.get("FH_VLM_API_KEY", "EMPTY")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def describe(self, image, prompt: str, max_tokens: int = 500) -> str:
        """image: PIL.Image。返回 VLM 的文字描述。"""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode())
        return out["choices"][0]["message"]["content"]


DESCRIBE_PROMPT = (
    "这是京津冀区域的{product}预报产品趋势拼图，各分格左上角标注预报时效。"
    "请用不超过150字概括：系统/量值的空间分布要点、随时效的演变趋势、"
    "对京津冀污染扩散条件的含义。只描述图中可见的事实，不要编造。{question}"
)
