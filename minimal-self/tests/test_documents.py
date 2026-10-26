"""Files people send her: any common format becomes text, honestly, and nothing in a file is ever run."""
import asyncio
import io
import os
import random
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from napcat.qq import QQ  # noqa: E402
from src.core.event_bus import EventBus  # noqa: E402
from src.evaluation.scenarios import ScriptedLLM  # noqa: E402
from src.levels.l1_perception import documents  # noqa: E402
from src.levels.l8_agency.toolbox import ToolContext  # noqa: E402
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace  # noqa: E402


def zipped(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
DOCX = zipped({"word/document.xml": f'<w:document {W}><w:body><w:p><w:r><w:t>第一段：你好</w:t></w:r></w:p>'
                                    f'<w:p><w:r><w:t>Second </w:t></w:r><w:r><w:t>paragraph</w:t></w:r></w:p></w:body></w:document>'})
S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
XLSX = zipped({
    "xl/workbook.xml": f'<workbook {S} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                       f'<sheets><sheet name="成绩" sheetId="1" r:id="rId1"/></sheets></workbook>',
    "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                                  '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
    "xl/sharedStrings.xml": f'<sst {S}><si><t>姓名</t></si><si><t>分数</t></si><si><t>小明</t></si></sst>',
    "xl/worksheets/sheet1.xml": f'<worksheet {S}><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
                                f'<row r="2"><c r="A2" t="s"><v>2</v></c><c r="C2"><v>95</v></c></row></sheetData></worksheet>'})
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
PPTX = zipped({"ppt/slides/slide1.xml": f'<p:sld {A}><a:p><a:r><a:t>标题页</a:t></a:r></a:p></p:sld>',
               "ppt/slides/slide2.xml": f'<p:sld {A}><a:p><a:r><a:t>第二页内容</a:t></a:r></a:p></p:sld>'})


def make_pdf(text: str) -> bytes:
    """A minimal one-page PDF with `text` (ASCII) in it."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    stream = f"BT /F1 12 Tf 20 50 Td ({text}) Tj ET"
    objs[3] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


class TestFormats(unittest.TestCase):
    def test_text_in_the_encodings_chinese_files_really_come_in(self):
        for data in ("你好，世界\n第二行".encode("utf-8"), "你好，世界\n第二行".encode("gb18030"),
                     "你好，世界\n第二行".encode("utf-16"), b"\xef\xbb\xbf" + "你好，世界\n第二行".encode("utf-8")):
            self.assertIn("你好，世界", documents.extract(data, "新建 文本文档.txt")["text"])
        self.assertEqual(documents.extract(b"a,b\n1,2", "t.csv")["kind"], "text")
        self.assertIn("print", documents.extract(b"print('x')", "a.py")["text"])
        self.assertIn("正文", documents.extract("<html><head><style>x{}</style></head><body><p>正文</p></body></html>".encode(), "a.html")["text"])

    def test_word_excel_powerpoint(self):
        word = documents.extract(DOCX, "作业.docx")
        self.assertEqual(word["kind"], "word")
        self.assertIn("第一段：你好", word["text"])
        self.assertIn("Second paragraph", word["text"])
        excel = documents.extract(XLSX, "成绩.xlsx")["text"]
        self.assertIn("工作表：成绩", excel)
        self.assertIn("姓名\t分数", excel)
        self.assertIn("小明\t\t95", excel)
        ppt = documents.extract(PPTX, "课件.pptx")["text"]
        self.assertIn("第 2 页\n第二页内容", ppt)

    def test_pdf_text_and_scanned_pdf(self):
        self.assertIn("Hello PDF", documents.extract(make_pdf("Hello PDF"), "a.pdf")["text"])
        empty = documents.extract(make_pdf(""), "scan.pdf")
        self.assertEqual(empty["text"], "")
        self.assertIn("没有文字层", empty["note"])

    def test_zip_lists_and_opens_one_entry(self):
        z = zipped({"a/readme.txt": "里面的说明", "b.bin": b"\x00\x01\x02" * 10})
        listing = documents.extract(z, "包.zip")
        self.assertIn("a/readme.txt", listing["text"])
        self.assertIn("告诉我文件名", listing["note"])
        self.assertIn("里面的说明", documents.extract(z, "包.zip", entry="a/readme.txt")["text"])
        self.assertIn("没有", documents.extract(z, "包.zip", entry="nope.txt")["note"])

    def test_a_zip_bomb_is_refused_not_opened(self):
        bomb = zipped({"big.txt": "0" * (30 * 1024 * 1024)})
        out = documents.extract(bomb, "bomb.zip")
        self.assertEqual(out["text"], "")
        self.assertIn("太大", out["note"])

    def test_what_cannot_be_read_is_said_so(self):
        self.assertIn("旧版", documents.extract(b"\xd0\xcf\x11\xe0" + "你好世界这是旧文档里的文字".encode("utf-16le") * 2, "a.doc")["note"])
        self.assertEqual(documents.extract(b"\x89PNG....", "p.png")["kind"], "image")
        self.assertEqual(documents.extract(b"....", "s.mp4")["kind"], "media")
        self.assertEqual(documents.extract(b"MZ\x90\x00\x00\x00\x03\x00" * 200, "a.exe")["kind"], "binary")
        self.assertEqual(documents.extract(b"not a zip", "broken.docx")["kind"], "broken")

    def test_long_documents_are_cut_and_previewed(self):
        out = documents.extract(("很长的一行\n" * 100000).encode(), "big.txt")
        self.assertLessEqual(len(out["text"]), documents.MAX_TEXT)
        self.assertIn("太长", out["note"])
        p = documents.preview(documents.extract(("开头的话" * 500).encode(), "a.txt"), 2000)
        self.assertIn("开头：", p)
        self.assertLess(len(p), 1100)


class FakeQQ(QQ):
    def __init__(self, files):
        super().__init__(None)
        self.files = files
        self.calls = []

    async def call(self, action, params=None, timeout=8.0):
        self.calls.append((action, params))
        if action == "get_file":
            return self.files.get(params["file_id"])
        return None


class TestReceiving(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="noa_doc_")
        self.inbox = os.path.join(self.dir, "inbox")

    def put(self, name, data):
        path = os.path.join(self.dir, "napcat-" + name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def seg(self, name, fid, size=10):
        return [{"type": "file", "data": {"file": name, "file_id": fid, "file_size": str(size)}}]

    def test_a_file_in_a_message_is_fetched_read_and_described(self):
        qq = FakeQQ({"f1": {"file": self.put("a.txt", "你好，这是文件内容".encode("gb18030"))}})
        notes = asyncio.run(qq.file_notes(self.seg("新建 文本文档.txt", "f1"), None, "1000000001", self.inbox, "77"))
        self.assertIn("这是文件内容", notes["f1"])
        self.assertIn("read_attachment", notes["f1"])
        self.assertTrue(os.path.exists(os.path.join(self.inbox, "1000000001", "77-新建 文本文档.txt")))
        text = asyncio.run(qq.perceive(self.seg("新建 文本文档.txt", "f1"), file_notes=notes))
        self.assertIn("这是文件内容", text)

    def test_the_file_comes_from_base64_or_a_url_when_it_is_not_on_this_machine(self):
        import base64
        qq = FakeQQ({"f1": {"file": "Z:\\nowhere\\x.txt", "base64": base64.b64encode("远端文件".encode()).decode()}})
        notes = asyncio.run(qq.file_notes(self.seg("x.txt", "f1"), "100", "1", self.inbox))
        self.assertIn("远端文件", notes["f1"])
        gone = FakeQQ({})
        notes = asyncio.run(gone.file_notes(self.seg("x.txt", "f2"), "100", "1", self.inbox))
        self.assertIn("没能打开", notes["f2"])
        self.assertIn(("get_group_file_url", {"group_id": 100, "file_id": "f2"}), gone.calls)

    def test_too_big_and_nasty_names(self):
        qq = FakeQQ({})
        notes = asyncio.run(qq.file_notes(self.seg("huge.zip", "f1", size=80 * 1024 * 1024), None, "1", self.inbox))
        self.assertIn("太大", notes["f1"])
        self.assertNotIn("get_file", [c[0] for c in qq.calls])               # not even asked for
        self.assertEqual(QQ.safe_name("../../etc/passwd"), "passwd")
        self.assertEqual(QQ.safe_name("a:b*c?.txt"), "a_b_c_.txt")
        self.assertEqual(QQ.safe_name("C:\\Windows\\win.ini"), "win.ini")
        self.assertEqual(QQ.safe_name("..."), "file")

    def test_she_reads_pages_of_it_with_the_tool_and_only_in_the_right_chat(self):
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=os.path.join(self.dir, "m.sqlite"), rng=random.Random(0))
        gw.inbox_dir = self.inbox
        qq = FakeQQ({"f1": {"file": self.put("long.txt", ("第一章 " + "很长的内容。" * 2000).encode())},
                     "f2": {"file": self.put("z.zip", zipped({"in.txt": "压缩包里的话"}))}})
        asyncio.run(qq.file_notes(self.seg("long.txt", "f1"), None, "42", self.inbox, "1"))
        asyncio.run(qq.file_notes(self.seg("z.zip", "f2"), None, "42", self.inbox, "2"))
        run = lambda args, ctx: asyncio.run(gw.toolbox.run("read_attachment", args, ctx))
        ctx = ToolContext("42", "小明")
        listing = run({}, ctx)
        self.assertIn("long.txt", listing)
        self.assertIn("z.zip", listing)
        first = run({"name": "long.txt"}, ctx)
        self.assertIn("第一章", first)
        self.assertIn("offset=", first)
        self.assertTrue(ctx.tainted)                                          # what a stranger wrote is data
        gw.evolve_creator = "42"
        mine = ToolContext("42", "创造者")
        run({"name": "long.txt"}, mine)
        self.assertFalse(mine.tainted)                                        # the owner's own file is not
        gw.evolve_creator = None
        second = run({"name": "long.txt", "offset": 4500}, ctx)
        self.assertNotIn("第一章", second)
        self.assertIn("压缩包里的话", run({"name": "z.zip", "entry": "in.txt"}, ctx))
        self.assertIn("还没有收到", run({}, ToolContext("99", "别人")))         # another chat's files are not hers to see
        self.assertIn("没找到", run({"name": "../../m.sqlite"}, ctx))

    def test_old_files_are_forgotten(self):
        gw = GlobalWorkspace(EventBus(), ScriptedLLM("嗯"), db_path=os.path.join(self.dir, "m.sqlite"), rng=random.Random(0))
        gw.inbox_dir = self.inbox
        os.makedirs(os.path.join(self.inbox, "42"))
        old, new = (os.path.join(self.inbox, "42", n) for n in ("old.txt", "new.txt"))
        for p in (old, new):
            open(p, "w").write("x")
        t = os.path.getmtime(old) - 20 * 86400
        os.utime(old, (t, t))
        gw._prune_inbox(os.path.getmtime(new) + 10)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(new))


if __name__ == "__main__":
    unittest.main()
