"""Recover complete model tool envelopes without turning their payload into speech."""
import json
import re
import uuid
import xml.etree.ElementTree as ET

MARKER = re.compile(r"DSML|<\s*/?\s*(?:tool_calls|function_calls|invoke|parameter)\b", re.I)
TOKEN = re.compile(r"<(/?)[｜|]+DSML[｜|]+\s*", re.I)


def protocol_text(text):
    return bool(MARKER.search(str(text or "")))


def spoken_text(text):
    # Even a truncated envelope may contain code, coordinates, or private arguments.
    # Suppress the entire response rather than guessing where an unfinished payload ends.
    return "" if protocol_text(text) else str(text or "")


def _valid(value, schema):
    kind = schema.get("type")
    types = {"string": str, "object": dict, "array": list, "boolean": bool,
             "integer": int, "number": (int, float)}
    if kind in types and (not isinstance(value, types[kind])
                          or kind in ("integer", "number") and isinstance(value, bool)):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            return False
        if any(k not in props for k in value) and schema.get("additionalProperties") is False:
            return False
        return all(_valid(v, props.get(k, {})) for k, v in value.items())
    if isinstance(value, list):
        return all(_valid(v, schema.get("items", {})) for v in value)
    return True


def recover_calls(content, specs):
    """Only a complete top-level calls envelope and currently offered tools are executable.

    The caller still uses Toolbox.run (permissions), and its normal step budget.
    Quoted examples and calls embedded in a spoken Response are intentionally not executed.
    """
    text = str(content or "").strip()
    if not protocol_text(text) or len(text) > 32000 or "<!" in text:
        return []
    text = TOKEN.sub(r"<\1", text)
    # Legacy model wrappers can precede a complete call. A quoted/fenced example cannot.
    prefix = text[:text.find("<calls")] if "<calls" in text else ""
    if prefix and not (re.match(r"^(?:Thought|Response)\s*[:：]", prefix)
                       and not any(mark in prefix for mark in ("```", "示例", "例如", "example", "引用"))):
        return []
    text = text[len(prefix):].strip()
    if not text.startswith("<calls"):
        return []
    try:
        root = ET.fromstring(text)
        if root.tag != "calls" or root.attrib or (root.text or "").strip() or not 1 <= len(root) <= 8:
            return []
        offered = {t["function"]["name"]: t["function"].get("parameters", {}) for t in specs or []}
        calls = []
        for invocation in root:
            name = invocation.get("name")
            if (invocation.tag != "invoke" or name not in offered
                    or set(invocation.attrib) != {"name"} or (invocation.text or "").strip()
                    or (invocation.tail or "").strip()):
                return []
            args = {}
            for parameter in invocation:
                key = parameter.get("name")
                if (parameter.tag != "parameter" or not key or key in args or len(parameter)
                        or set(parameter.attrib) - {"name", "string"} or (parameter.tail or "").strip()
                        or parameter.get("string") not in ("true", "false")):
                    return []
                value = parameter.text or ""
                args[key] = value if parameter.get("string") == "true" else json.loads(value)
            if not _valid(args, offered[name]):
                return []
            calls.append({"id": "recovered_" + uuid.uuid4().hex, "name": name, "arguments": args})
        return calls
    except (ET.ParseError, ValueError, TypeError, KeyError):
        return []


def split_calls(content, specs):
    """A reply that talks first and then writes its calls out as text ("Response: 插个火把 <calls>...</calls>").

    Returns (the prose before the envelope, the calls) when the trailing envelope is complete and valid by the same
    rules as recover_calls; otherwise (content, []). Only for a caller that acts with her own body (the game), where
    the alternative is saying she does something and then not doing it.
    """
    text = str(content or "")
    if not protocol_text(text) or len(text) > 32000:
        return text, []
    normal = TOKEN.sub(r"<\1", text)
    start, end = normal.find("<calls"), normal.rfind("</calls>")
    if start < 0 or end < start or normal[end + len("</calls>"):].strip():
        return text, []
    calls = recover_calls(normal[start:end + len("</calls>")], specs)
    if not calls:
        return text, []
    return normal[:start].rstrip(), calls


def assistant_calls(content, calls):
    return {"role": "assistant", "content": spoken_text(content) or None,
            "tool_calls": [{"id": c["id"], "type": "function", "function": {
                "name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in calls]}
