"""
Writes the architecture diagrams used by docs/ARCHITECTURE.md as standalone SVG files (no dependencies).

    python scripts/make_architecture_diagrams.py

Four diagrams: overview, level map (L1-L19), path of one message, permissions and firewall.
Edit the boxes below when the architecture changes, then run this and commit docs/architecture-*.svg.
"""
import os
from xml.sax.saxutils import escape

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")

STYLE = """<style>
text{font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",sans-serif}
.t{font-size:14px;fill:#2C2C2A}.ts{font-size:12px;fill:#5F5E5A}.th{font-size:14px;font-weight:500;fill:#2C2C2A}
.box{fill:#FFFFFF;stroke:#B4B2A9}
.arr{stroke:#5F5E5A;stroke-width:1.5;fill:none}
.c-gray>rect{fill:#F1EFE8;stroke:#888780}.c-gray .th{fill:#444441}.c-gray .ts{fill:#5F5E5A}
.c-purple>rect{fill:#EEEDFE;stroke:#534AB7}.c-purple .th{fill:#3C3489}.c-purple .ts{fill:#534AB7}
.c-teal>rect{fill:#E1F5EE;stroke:#0F6E56}.c-teal .th{fill:#085041}.c-teal .ts{fill:#0F6E56}
</style>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#5F5E5A" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>"""


class Svg:
    def __init__(self, height: int, title: str, desc: str):
        self.height, self.title, self.desc = height, title, desc
        self.parts = []

    def box(self, x, y, w, h, title, sub=None, sub2=None, ramp="gray", plain=False):
        cx = x + w / 2
        cls = "" if plain else f' class="c-{ramp}"'
        rect = f'<rect{" class=" + chr(34) + "box" + chr(34) if plain else ""} x="{x}" y="{y}" width="{w}" height="{h}" rx="8" stroke-width="0.5"/>'
        lines = [t for t in (sub, sub2) if t]
        if not lines:
            texts = f'<text class="th" x="{cx}" y="{y + h / 2}" text-anchor="middle" dominant-baseline="central">{escape(title)}</text>'
        else:
            top = y + h / 2 - 10 * len(lines)
            texts = f'<text class="th" x="{cx}" y="{top + 1}" text-anchor="middle" dominant-baseline="central">{escape(title)}</text>'
            for i, line in enumerate(lines):
                texts += f'<text class="ts" x="{cx}" y="{top + 21 + 19 * i}" text-anchor="middle" dominant-baseline="central">{escape(line)}</text>'
        self.parts.append(f"<g{cls}>{rect}{texts}</g>")

    def row(self, x, y, w, left, right, ramp, lw=150, h=38):
        """Level-map row: a name box and a description box."""
        self.parts.append(f'<g class="c-{ramp}"><rect x="{x}" y="{y}" width="{lw}" height="{h}" rx="6" stroke-width="0.5"/>'
                          f'<text class="th" x="{x + lw / 2}" y="{y + h / 2}" text-anchor="middle" dominant-baseline="central">{escape(left)}</text></g>')
        self.parts.append(f'<g class="c-{ramp}"><rect x="{x + lw + 10}" y="{y}" width="{w - lw - 10}" height="{h}" rx="6" stroke-width="0.5"/>'
                          f'<text class="ts" x="{x + lw + 26}" y="{y + h / 2}" dominant-baseline="central">{escape(right)}</text></g>')

    def container(self, x, y, w, h, title, ramp="purple"):
        self.parts.append(f'<g class="c-{ramp}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" stroke-width="0.5"/>'
                          f'<text class="th" x="{x + 20}" y="{y + 22}" dominant-baseline="central">{escape(title)}</text></g>')

    def arrow(self, x1, y1, x2, y2):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="arr" marker-end="url(#arrow)"/>')

    def bend(self, d):
        self.parts.append(f'<path d="{d}" class="arr" fill="none" marker-end="url(#arrow)"/>')

    def text(self, x, y, s, anchor="start", cls="ts"):
        self.parts.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="central">{escape(s)}</text>')

    def dashed(self, x, y, w, h):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="4 4"/>')

    def save(self, name):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="{self.height}" viewBox="0 0 680 {self.height}" role="img">'
               f'<title>{escape(self.title)}</title><desc>{escape(self.desc)}</desc>\n{STYLE}\n'
               f'<rect width="680" height="{self.height}" fill="#FFFFFF"/>\n{body}\n</svg>\n')
        path = os.path.join(DOCS, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(svg)
        print("wrote", os.path.relpath(path))


def overview():
    s = Svg(630, "乃爱整体架构", "输入层经感知进入 GlobalWorkspace 心智，再用模型、受权限与防火墙约束的外部工具和自我改进通道，全部落在一个记忆库上。")
    for x, t, sub in ((40, "QQ · NapCat", "私聊 群 文件"), (190, "控制台", "监控 开关"), (340, "Minecraft", "游戏聊天"), (490, "时间 · 身体", "心跳 睡眠")):
        s.box(x, 26, 130, 50, t, sub)
        s.arrow(x + 65, 76, x + 65, 112)
    s.box(40, 116, 580, 50, "app.py 路由 + napcat/qq.py 感知", "引用 转发 图片 语音 文件 都变成文字")
    s.arrow(330, 166, 330, 202)
    s.container(40, 206, 580, 214, "GlobalWorkspace（L10）：她的心智")
    for x, t, sub in ((60, "记忆 L2", "事件 事实 关系"), (200, "感受 L3-7", "情绪 需要 价值"), (340, "自我 L9-10", "叙事 人格 元认知"), (480, "行动 L8", "工具 目标 任务")):
        s.box(x, 246, 120, 66, t, sub, plain=True)
    s.box(60, 332, 540, 68, "一条消息的路：听 理解 回忆 决定 说", "潜台词 · 提示词 · 工具循环 · 记成经历", plain=True)
    for x in (130, 340, 540):
        s.arrow(x, 420, x, 456)
        s.arrow(x, 516, x, 546)
    s.box(40, 460, 180, 56, "DeepSeek 模型", "工具调用 看图")
    s.box(240, 460, 200, 56, "外部工具 MCP", "权限分层 + 防火墙", ramp="teal")
    s.box(460, 460, 160, 56, "自我改进 L17", "DSH 维护请求")
    s.box(40, 550, 580, 50, "memory.sqlite（39 表）· 每日备份 · 状态快照", "工作文件夹：收件箱 浏览器资料 脚本")
    s.save("architecture-1-overview.svg")


def level_map():
    s = Svg(650, "层级地图 L1 到 L19", "每一层和它的主要模块：底座、感知、记忆、代谢、情绪、价值、稳态、理解人、行动、叙事、意识、自我改进、知识与搜索。")
    rows = [("core 底座", "时钟（可模拟）· 事件 · 状态快照 · 每日备份", "gray"),
            ("L1 感知", "感官 · 环境 · 群语料 · 表情库 · 文件读成文字", "gray"),
            ("L2 记忆", "本地向量检索 · 事件/事实 · 关系图 · 对话线 · 人物画像", "gray"),
            ("L3 代谢", "精力 · 睡眠恢复 · 累了说话变慢变短", "purple"),
            ("L4 情绪", "六种情绪 · 互相压制 · 习惯化 · 心境 · 睡觉安抚", "purple"),
            ("L5 价值", "七个价值 · 冲突与拒绝 · 随经历慢慢变", "purple"),
            ("L6 稳态", "饥饿 · 孤独 · 好奇 · 卫生，随时间增长", "purple"),
            ("L7 理解人", "预测误差 · 读潜台词 · 传闻与敏感信息", "purple"),
            ("L8 行动", "工具箱 · MCP · 权限 · 防火墙 · 漫游 · 游戏 · 目标", "teal"),
            ("L9 叙事", "日记自传 · 人格漂移 · 说话范例", "purple"),
            ("L10 意识", "总编排 · 注意力图式 · 元认知 · 内心独白", "purple"),
            ("L17 改进", "自诊断 · 适应度评估 · DSH · 维护请求", "teal"),
            ("L18-19 知识", "知识库 · 三级回退搜索（没查到就明说）", "gray")]
    for i, (name, desc, ramp) in enumerate(rows):
        s.row(40, 20 + 46 * i, 600, name, desc, ramp)
    s.text(40, 626, "紫：内在状态　青：对外行动与改进　灰：底座与知识")
    s.save("architecture-2-levels.svg")


def message_path():
    s = Svg(480, "一条消息的路径", "从 QQ 事件到回复被记成经历的九步：感知、判断要不要回、读潜台词、感受与回忆、拼提示词、带权限与防火墙的工具循环、落地、记忆。")
    left = [("1 事件进来", "去重 · 限流", "gray"), ("2 感知翻译", "引用 图片 语音 文件读成文字", "gray"),
            ("3 要不要管", "睡着 静音 限流 插话", "gray"), ("4 读潜台词", "只对约 5% 的消息", "purple"),
            ("5 感受与回忆", "情绪 预测误差 记忆召回", "purple")]
    for i, (t, sub, ramp) in enumerate(left):
        y = 30 + 86 * i
        s.box(40, y, 270, 56, t, sub, ramp=ramp)
        if i < len(left) - 1:
            s.arrow(175, y + 56, 175, y + 82)
    s.bend("M310 402 L340 402 L340 58 L370 58")
    s.box(370, 30, 270, 56, "6 拼提示词", "稳定的在前，利于缓存")
    s.arrow(505, 86, 505, 112)
    s.box(370, 116, 270, 86, "7 工具循环", "权限档 → 防火墙 → 执行", "→ 扫描结果 → 隐藏密钥", ramp="teal")
    s.arrow(505, 202, 505, 228)
    s.box(370, 232, 270, 56, "8 落地", "清洗 分条 打字延迟")
    s.arrow(505, 288, 505, 314)
    s.box(370, 318, 270, 56, "9 记成经历", "对话 状态 因果 决策留痕")
    s.dashed(370, 402, 270, 56)
    s.text(505, 421, "沉默是合法的结果：", anchor="middle")
    s.text(505, 441, "3 或 6 之后她可以选择不说", anchor="middle")
    s.save("architecture-3-message-path.svg")


def permissions():
    s = Svg(480, "权限与防火墙", "五类提出要求的人经过同一道防火墙的五项检查，才能用到浏览器、文件与脚本、或游戏。")
    askers = [("创造者", "配置的 QQ 号 · 控制台"), ("受信任的人", "名单内 · 私聊"), ("她自己", "漫游 · 游戏自玩"),
              ("游戏玩家", "名单内"), ("其他人和群", "每天限次")]
    for i, (t, sub) in enumerate(askers):
        y = 40 + 72 * i
        s.box(30, y, 150, 56, t, sub)
        s.arrow(180, y + 28, 236, y + 28)
    s.container(240, 40, 200, 344, "防火墙", ramp="teal")
    for i, line in enumerate(("1 扫描网页和文件里的命令", "2 代码先检查再关进守卫", "3 她自己的端口不能碰", "4 保护记忆库和密钥", "5 说出口的话抹掉密钥")):
        s.text(340, 104 + 42 * i, line, anchor="middle")
    s.text(340, 330, "命中就整轮只读", anchor="middle")
    s.text(340, 352, "紧急封锁一键停用", anchor="middle")
    for i, (t, sub, sub2) in enumerate((("浏览器", "看 填表 上传", "按档位放开"), ("文件与脚本", "读写 运行 Python", "只给创造者"), ("Minecraft", "走 挖 放 聊天", "默认关闭"))):
        y = 70 + 102 * i
        s.arrow(440, y + 40, 496, y + 40)
        s.box(500, y, 150, 80, t, sub, sub2=sub2)
    s.text(40, 430, "创造者对脚本和文件有信任，但识别到注入的那一轮也只读。", anchor="start")
    s.text(40, 452, "防火墙是防护栏，不是真沙箱，所以脚本只给创造者。", anchor="start")
    s.save("architecture-4-permissions.svg")


if __name__ == "__main__":
    for make in (overview, level_map, message_path, permissions):
        make()
