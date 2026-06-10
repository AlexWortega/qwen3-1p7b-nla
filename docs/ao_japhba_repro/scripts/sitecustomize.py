# Auto-loaded at interpreter startup (repo root on sys.path).
# Routes the Anthropic SDK used by the eval judges through OpenRouter.
import os
def _patch_anthropic():
    try:
        import anthropic
        from openai import AsyncOpenAI, OpenAI
    except Exception:
        return
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return
    base = "https://openrouter.ai/api/v1"
    def map_model(m):
        ml = (m or "").lower()
        if "haiku" in ml:  return "anthropic/claude-haiku-4.5"
        if "sonnet" in ml: return "anthropic/claude-sonnet-4.5"
        if "opus" in ml:   return "anthropic/claude-opus-4.1"
        return m
    class _Blk:
        def __init__(self, text): self.text = text
    class _Resp:
        def __init__(self, text): self.content = [_Blk(text)]
    def _conv(system, messages):
        out = []
        if system: out.append({"role": "system", "content": system})
        for msg in (messages or []):
            c = msg.get("content")
            if isinstance(c, list):
                c = "".join(b.get("text","") if isinstance(b,dict) else str(b) for b in c)
            out.append({"role": msg["role"], "content": c})
        return out
    class _AMsgs:
        def __init__(self, c): self._c = c
        async def create(self, *, model, max_tokens=1024, system=None, messages=None, **kw):
            r = await self._c.chat.completions.create(model=map_model(model), max_tokens=max_tokens,
                messages=_conv(system, messages), temperature=kw.get("temperature",0.0))
            return _Resp(r.choices[0].message.content or "")
    class _SMsgs:
        def __init__(self, c): self._c = c
        def create(self, *, model, max_tokens=1024, system=None, messages=None, **kw):
            r = self._c.chat.completions.create(model=map_model(model), max_tokens=max_tokens,
                messages=_conv(system, messages), temperature=kw.get("temperature",0.0))
            return _Resp(r.choices[0].message.content or "")
    class ShimAsync:
        def __init__(self, *a, **k): self.messages = _AMsgs(AsyncOpenAI(base_url=base, api_key=key))
    class ShimSync:
        def __init__(self, *a, **k): self.messages = _SMsgs(OpenAI(base_url=base, api_key=key))
    anthropic.AsyncAnthropic = ShimAsync
    anthropic.Anthropic = ShimSync
_patch_anthropic()
