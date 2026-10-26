"""
Level 1d: reading files people send her - text, PDF, Word, Excel, PowerPoint, zip, e-books.

Only the standard library plus pypdf (for PDF). Office files are zip archives of XML, so they are read
directly: paragraphs of a .docx, cells of a .xlsx (as tab-separated rows), slide text of a .pptx. Nothing in
a file is ever executed: it is turned into text and nothing else, with limits on size and on how much a
compressed file may expand (a zip bomb is just refused).

`extract()` always returns a dict and never raises: {"kind", "text", "note"} - `note` says what could not
be done ("扫描件，没有文字层", "旧版 .doc 只能粗略提取") so she can tell the person the truth.
"""
import io
import os
import re
import zipfile
from html import unescape
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

MAX_MEMBER = 25 * 1024 * 1024          # one file inside an archive, uncompressed
MAX_TEXT = 400_000                     # characters kept of a document
MAX_PAGES = 60
MAX_ROWS = 300

TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".xml", ".log", ".ini", ".cfg",
            ".conf", ".toml", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs",
            ".rb", ".php", ".sh", ".bat", ".ps1", ".sql", ".srt", ".vtt", ".tex", ".css", ".vue", ".kt", ".swift", ".lua",
            ".r", ".env", ".gitignore", ".properties", ".lrc", ".ass", ".rst", ".bib"}
HTML_EXT = {".html", ".htm", ".xhtml", ".mht"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
AV_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".amr", ".silk"}

def _decode(data: bytes) -> str:
    """Text in whatever encoding Chinese files really come in."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    for enc in ("utf-8", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")

def _looks_like_text(data: bytes) -> bool:
    sample = data[:4000]
    if b"\x00" in sample and not sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return False
    bad = sum(1 for b in sample if b < 9 or 13 < b < 32)
    return bad < max(3, len(sample) * 0.02)

def _html_text(html: str) -> str:
    body = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", html)
    body = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>|</h\d>", "\n", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = unescape(body)
    body = re.sub(r"[ \t\r\f\v]+", " ", body)
    return re.sub(r"\n\s*\n+", "\n", body).strip()

# --- Office / archives -------------------------------------------------------------------

def _zip(data: bytes) -> zipfile.ZipFile:
    z = zipfile.ZipFile(io.BytesIO(data))
    total = sum(i.file_size for i in z.infolist())
    if total > 8 * MAX_MEMBER or any(i.file_size > MAX_MEMBER for i in z.infolist()):
        raise ValueError("压缩包解开后太大，不读（可能是压缩炸弹）")
    return z

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]

def _docx(data: bytes) -> str:
    z = _zip(data)
    root = ET.fromstring(z.read("word/document.xml"))
    lines: List[str] = []
    for p in root.iter():
        if _local(p.tag) != "p":
            continue
        text = "".join((t.text or "") for t in p.iter() if _local(t.tag) == "t")
        if text.strip():
            lines.append(text)
    return "\n".join(lines)

def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref or "")
    n = 0
    for ch in letters.group(0) if letters else "A":
        n = n * 26 + ord(ch) - 64
    return n - 1

def _xlsx(data: bytes) -> str:
    z = _zip(data)
    shared: List[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root:
            shared.append("".join((t.text or "") for t in si.iter() if _local(t.tag) == "t"))
    names = {}
    if "xl/workbook.xml" in z.namelist():
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = {}
        if "xl/_rels/workbook.xml.rels" in z.namelist():
            for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")):
                rels[r.get("Id")] = r.get("Target")
        for s in wb.iter():
            if _local(s.tag) == "sheet":
                rid = next((v for k, v in s.attrib.items() if k.endswith("}id")), None)
                target = rels.get(rid, "")
                path = target.lstrip("/") if target.startswith("/") else "xl/" + target
                names[path] = s.get("name")
    out: List[str] = []
    sheets = sorted((n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)),
                    key=lambda n: int(re.findall(r"\d+", n)[-1]))
    for path in sheets:
        out.append(f"## 工作表：{names.get(path, os.path.basename(path))}")
        root = ET.fromstring(z.read(path))
        rows = 0
        for row in root.iter():
            if _local(row.tag) != "row":
                continue
            cells: Dict[int, str] = {}
            for c in row:
                if _local(c.tag) != "c":
                    continue
                kind, value = c.get("t"), ""
                v = next((x for x in c if _local(x.tag) == "v"), None)
                if kind == "s" and v is not None and (v.text or "").isdigit():
                    idx = int(v.text)
                    value = shared[idx] if idx < len(shared) else ""
                elif kind == "inlineStr":
                    value = "".join((t.text or "") for t in c.iter() if _local(t.tag) == "t")
                elif v is not None:
                    value = v.text or ""
                cells[_col_index(c.get("r"))] = value
            if cells:
                out.append("\t".join(cells.get(i, "") for i in range(max(cells) + 1)).rstrip())
                rows += 1
                if rows >= MAX_ROWS:
                    out.append(f"……（这个工作表只显示前 {MAX_ROWS} 行）")
                    break
    return "\n".join(out)

def _pptx(data: bytes) -> str:
    z = _zip(data)
    slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                    key=lambda n: int(re.findall(r"\d+", n)[-1]))
    out = []
    for i, path in enumerate(slides, 1):
        root = ET.fromstring(z.read(path))
        paras = []
        for p in root.iter():
            if _local(p.tag) == "p":
                text = "".join((t.text or "") for t in p.iter() if _local(t.tag) == "t")
                if text.strip():
                    paras.append(text)
        out.append(f"## 第 {i} 页\n" + "\n".join(paras))
    return "\n\n".join(out)

def _epub(data: bytes) -> str:
    z = _zip(data)
    pages = sorted(n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm")))
    return "\n\n".join(_html_text(_decode(z.read(n))) for n in pages[:80])

def _rtf(data: bytes) -> str:
    text = _decode(data)
    text = re.sub(r"\\'([0-9a-f]{2})", lambda m: bytes([int(m.group(1), 16)]).decode("gb18030", errors="replace"), text)
    text = re.sub(r"\\u(-?\d+)\??", lambda m: chr(int(m.group(1)) % 65536), text)
    text = re.sub(r"\\[a-z]+-?\d* ?|[{}]", "", text)
    return text.strip()

def _pdf(data: bytes) -> Dict[str, Any]:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return {"text": "", "note": "这个 PDF 有密码，打不开"}
    pages, out = len(reader.pages), []
    for i, page in enumerate(reader.pages[:MAX_PAGES], 1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append(f"## 第 {i} 页\n{text}")
    note = f"共 {pages} 页" + (f"，只读了前 {MAX_PAGES} 页" if pages > MAX_PAGES else "")
    if not out:
        note = "这个 PDF 里没有文字层（多半是扫描件或图片），读不出文字"
    return {"text": "\n\n".join(out), "note": note}

def _legacy_strings(data: bytes) -> str:
    """Old binary Word/Excel/PowerPoint: no structure, but the readable runs of text can be pulled out."""
    runs = re.findall(rb"(?:[\x20-\x7e]\x00){6,}", data)
    text = [r.decode("utf-16le", errors="ignore") for r in runs]
    runs8 = re.findall(rb"[\x20-\x7e]{12,}", data)
    text += [r.decode("ascii") for r in runs8]
    seen, out = set(), []
    for t in text:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return "\n".join(out)

def zip_entries(data: bytes) -> List[Dict[str, Any]]:
    z = zipfile.ZipFile(io.BytesIO(data))
    return [{"name": i.filename, "size": i.file_size} for i in z.infolist() if not i.is_dir()][:200]

# --- the front door ----------------------------------------------------------------------------

def extract(data: bytes, name: str, entry: str = None) -> Dict[str, Any]:
    """Text of a file. `entry` picks one file inside a zip. Never raises."""
    ext = os.path.splitext(name.lower())[1]
    try:
        if ext == ".zip" or (entry and zipfile.is_zipfile(io.BytesIO(data)) and ext not in (".docx", ".xlsx", ".pptx", ".epub")):
            z = _zip(data)
            if entry:
                if entry not in z.namelist():
                    return {"kind": "zip", "text": "", "note": f"压缩包里没有 {entry}"}
                return extract(z.read(entry), entry)
            listing = "\n".join(f"{e['name']}（{e['size']} 字节）" for e in zip_entries(data))
            return {"kind": "zip", "text": "压缩包里的文件：\n" + listing,
                    "note": "这是压缩包；要看里面哪个文件，告诉我文件名"}
        if ext == ".pdf" or data[:5] == b"%PDF-":
            r = _pdf(data)
            return {"kind": "pdf", **_cut(r["text"]), "note": r["note"] + _cut(r["text"])["note"]}
        if ext == ".docx":
            return {"kind": "word", **_cut(_docx(data))}
        if ext == ".xlsx" or ext == ".xlsm":
            return {"kind": "excel", **_cut(_xlsx(data))}
        if ext == ".pptx":
            return {"kind": "powerpoint", **_cut(_pptx(data))}
        if ext == ".epub":
            return {"kind": "ebook", **_cut(_epub(data))}
        if ext == ".rtf":
            return {"kind": "rtf", **_cut(_rtf(data))}
        if ext in (".doc", ".xls", ".ppt", ".wps", ".et", ".dps"):
            r = _cut(_legacy_strings(data))
            return {"kind": "office-old", "text": r["text"],
                    "note": "这是旧版格式（" + ext + "），只能粗略提取出能读的文字，排版和表格会乱；最好让对方另存为新版格式再发" + r["note"]}
        if ext in HTML_EXT:
            return {"kind": "html", **_cut(_html_text(_decode(data)))}
        if ext in IMAGE_EXT:
            return {"kind": "image", "text": "", "note": "这是一张图片文件，读文字要用图片识别"}
        if ext in AV_EXT:
            return {"kind": "media", "text": "", "note": "这是音频或视频，我现在读不了里面的内容"}
        if ext in TEXT_EXT or _looks_like_text(data):
            return {"kind": "text", **_cut(_decode(data))}
        return {"kind": "binary", "text": "", "note": f"不认识这种文件（{ext or '没有扩展名'}），读不了"}
    except zipfile.BadZipFile:
        return {"kind": "broken", "text": "", "note": "文件是坏的，或者不是它名字说的那种格式"}
    except ValueError as e:
        return {"kind": "refused", "text": "", "note": str(e)}
    except Exception as e:
        return {"kind": "error", "text": "", "note": f"读取出错了：{type(e).__name__}: {str(e)[:100]}"}

def _cut(text: str) -> Dict[str, str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > MAX_TEXT:
        return {"text": text[:MAX_TEXT], "note": f"；太长，只保留了前 {MAX_TEXT} 个字"}
    return {"text": text, "note": ""}

def preview(result: Dict[str, Any], size: int, limit: int = 600) -> str:
    """How a received file shows up in what she reads: what it is, and the beginning."""
    text = (result.get("text") or "").strip()
    head = text[:limit] + ("……" if len(text) > limit else "")
    note = f"（{result['note'].lstrip('；')}）" if result.get("note") else ""
    return f"{size_words(size)}，{result['kind']}{note}" + (f"。开头：\n{head}" if head else "")

def size_words(n: int) -> str:
    return f"{n} 字节" if n < 1024 else f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"
