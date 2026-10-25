"""Generate the architecture diagrams for Noa (docs/*.png).

    python scripts/make_architecture_diagrams.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
os.makedirs(OUT, exist_ok=True)

INK, GREY = "#111827", "#6b7280"
LINE = "#64748b"

P = {
    "ext": ("#e0f2fe", "#0284c7"),
    "io": ("#fef3c7", "#b45309"),
    "perc": ("#ede9fe", "#6d28d9"),
    "state": ("#fee2e2", "#b91c1c"),
    "cog": ("#dcfce7", "#15803d"),
    "ws": ("#dbeafe", "#1d4ed8"),
    "llm": ("#fae8ff", "#a21caf"),
    "evo": ("#fef9c3", "#a16207"),
    "data": ("#f1f5f9", "#334155"),
}


def panel(ax, x, y, w, h, title, key):
    fc, ec = P[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.45,rounding_size=1.0",
                                facecolor=fc, edgecolor=ec, linewidth=1.5, alpha=0.5, zorder=1))
    ax.text(x + 0.9, y + h - 0.9, title, fontsize=11, fontweight="bold", color=ec, va="top", ha="left", zorder=6)


def box(ax, x, y, w, h, title, lines=(), key="data", fs=7.4, tfs=8.4, fill=None):
    fc, ec = P[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.22,rounding_size=0.7",
                                facecolor=fill or "#ffffff", edgecolor=ec, linewidth=1.1, zorder=3))
    ax.text(x + w / 2, y + h - 0.75, title, fontsize=tfs, fontweight="bold", color=INK,
            ha="center", va="top", zorder=6)
    if lines:
        ax.text(x + w / 2, y + h - 2.25, "\n".join(lines), fontsize=fs, color="#374151",
                ha="center", va="top", linespacing=1.5, zorder=6)


def arrow(ax, p1, p2, color=None, lw=1.6, rad=0.0, label=None, fs=7.0, ls="-", style="-|>"):
    color = color or LINE
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, linewidth=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}", zorder=2,
                                 linestyle=ls, shrinkA=3, shrinkB=3))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, fontsize=fs, color=color, ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))


# ---------------------------------------------------------------------------
def structure():
    fig, ax = plt.subplots(figsize=(18.4, 12.6), dpi=150)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 84)
    ax.axis("off")

    ax.text(2, 82.4, "乃爱（Noa）· 架构全貌", fontsize=20, fontweight="bold", color=INK)
    ax.text(2, 80.3, "数字生命实验｜单进程 Python（状态机 + 提示词拼装器）+ 三个外部小脑（NapCat / 语音 / 视觉）｜2026-10-25",
            fontsize=9, color=GREY)

    # ================= column 1: outside =================
    panel(ax, 1.5, 29.5, 31.5, 48.5, "① 外部接入 · 感知 · 输出", "ext")
    box(ax, 3.0, 68.0, 28.5, 7.0, "QQ 用户 / 群",
        ["私聊 · 群聊 · @ · 引用 · 图片 · 语音", "文件 · 合并转发 · 卡片 · 戳一戳"], "ext")
    box(ax, 3.0, 59.0, 28.5, 7.2, "NapCat 4.18.41 + QQ NT",
        ["OneBot v11 反向 WebSocket", "ws://127.0.0.1:8080/onebot/v11/ws/noa", "OCR / 语音转码 / 取图它都有"], "ext")
    box(ax, 3.0, 49.0, 28.5, 8.2, "napcat/ 感知适配",
        ["napcat_server：事件分发 · echo 与响应", "qq.py：perceive() 消息段 → 人看到的样子", "notice()：戳/撤回/进群/禁言/表情", "senses.py：听觉 · 视觉"], "io")
    box(ax, 3.0, 39.5, 28.5, 8.2, "L1 感知",
        ["sensory：痛 / 触摸 / 喂食", "environment：昼夜 / 天气 / 节日", "group_language：群语料 / 梗", "stickers：表情库 + 使用语境"], "perc")
    box(ax, 3.0, 31.5, 28.5, 6.4, "能力自查（真实）",
        ["ocr=读图上的字 · hear=听语音", "see=看懂图片 · 缺失她就说自己看不见"], "perc", fill="#faf5ff")


    # ================= column 2: inside =================
    panel(ax, 34.5, 29.5, 52.0, 48.5, "② 内部世界：GlobalWorkspace（唯一的「我」）", "ws")
    box(ax, 36.5, 68.6, 48.0, 5.8, "L10 编排核心（2,479 行）",
        ["tick() 每 10 秒一拍 ≈ 活着  ·  hear() 处理每条消息  ·  一个事件循环 / 一个 SQLite / 一个闭环"], "ws", fs=7.2)
    box(ax, 36.5, 61.0, 23.2, 7.6, "工作区竞争 + 元认知",
        ["Workspace：候选打分 → 取 2 个胜者", "AttentionSchema：我自己的注意力模型", "Metacognition：置信度 / 校准"], "ws", fill="#eff6ff")
    box(ax, 61.3, 61.0, 23.2, 7.6, "Prompt 组装（此刻的她）",
        ["人格 · 自传 · 连续性 · 心里在想", "身体心情 · 记忆 · 群语料 · 表情", "能力边界 · 关系/ToM · 工具"], "ws", fill="#eff6ff")
    box(ax, 36.5, 50.5, 23.2, 8.4, "L2 记忆（分区）",
        ["episodes 情景（4KB embedding/条）", "semantic_facts 带引用的事实", "autobiography 自传", "Embedder = 字符 n-gram 哈希"], "cog")
    box(ax, 61.3, 50.5, 23.2, 8.4, "L7 心智理论 · L9 叙事",
        ["user_model：延迟/回复率/情感/像机器人", "narrative(dream)：日记 · 自传 · 事实", "trait drift：人格缓慢漂移（有账本）"], "cog")
    box(ax, 36.5, 40.5, 23.2, 8.2, "L8 驱动力 / 目标",
        ["Q 学习：need×Q−成本−疲劳 → softmax", "goal：承诺 / 关系 / 学习 / 自我", "自主找人 · 搜索 · 插话 · 沉默"], "cog")
    box(ax, 61.3, 40.5, 23.2, 8.2, "L3~L6 身体 · L5 价值",
        ["精力/睡眠 · 6 维情绪 · 饿/孤独/好奇", "7 价值：尊严/真诚/亲近/好奇/", "          自主/安稳/公平", "价值演化 + 拒绝机制"], "state")
    box(ax, 36.5, 31.5, 48.0, 7.2, "L14 意识流 · L15 存在焦虑 · L16 人格稳态",
        ["内心独白（赢得工作区才进记忆）· 焦虑（断线/威胁/安慰/梦）· 气质不能一夜被改写"], "#fff1f2" and "state", fs=7.2)

    # ================= column 3: generation =================
    panel(ax, 88.5, 29.5, 30.0, 48.5, "③ 语言 · 工具 · 改造", "llm")
    box(ax, 90.0, 67.4, 27.0, 7.6, "LLM 引擎（可换）",
        ["DeepSeekEngine（当前 deepseek-flash）", "HybridEngine → LoRA :8093（她自己的模型）", "ChatML 与 messages 适配"], "llm")
    box(ax, 90.0, 60.5, 27.0, 7.4, "输出协议",
        ["Thought / Response 分离", "（不回）= 真的沉默", "Action: RECALL/MEMORIZE/SEARCH/EVOLVE"], "llm")
    box(ax, 90.0, 51.5, 27.0, 7.4, "工具",
        ["L19 搜索：ddgs→百度→DDG HTML", "L18 知识库：标题/正文/标签", "L17 代码解释器：已关闭"], "llm")
    box(ax, 90.0, 40.5, 27.0, 9.4, "自我改造通道（DSH）",
        ["EVOLVE → DSH ACP 子进程", "一次性副本里改代码，只回传 diff", "补丁 ≤300 行 · 每天 1 次 · 需审核", "apply 后重启生效，可回滚"], "evo")
    box(ax, 90.0, 31.5, 27.0, 7.6, "触发器（不靠模型自觉）",
        ["本地识别「改自己」意图 → 直接进通道", "!evolve / 自诊断 / 6 小时自检"], "evo", fill="#fefce8")

    # ================= bottom band =================
    panel(ax, 1.5, 2.0, 117.0, 26.0, "④ 自我改造闭环 · 数据 · 运维", "evo")
    loop = [
        ("① 自诊断", ["从真实记录里挑问题：", "校准偏差 / 没兑现的承诺 /", "反刍 / 搜索失败"]),
        ("② 适应度", ["16 个行为场景 × 多 seed", "子进程跑候选代码（不导入线上）", "硬约束：原通过的不能挂"]),
        ("③ DSH 提案", ["ACP：initialize → session", "→ 副本改代码 → 跑测试", "→ 只带回补丁 + 结果"]),
        ("④ 审核 → 生效", ["scripts/evolve.py apply", "或控制台按钮 → 重启", "7 天后回读真实指标"]),
    ]
    for i, (t, lines) in enumerate(loop):
        x = 3.5 + i * 22.0
        box(ax, x, 17.4, 20.4, 7.2, t, lines, "evo", fill="#fffbeb", fs=7.1)
        if i < 3:
            arrow(ax, (x + 20.4, 21.0), (x + 22.0, 21.0), color=P["evo"][1], lw=1.5)
    box(ax, 3.5, 4.6, 27.6, 11.0, "memory.sqlite（一个文件＝她的一生）",
        ["users / interactions / episodes", "semantic_facts / memories / relationships",
         "persona_traits / trait_changes / self_state", "state_causes / q_values / user_models",
         "stickers / group_phrases / goals", "self_improvement / evolution_proposals"], "data", fs=7.0)
    box(ax, 32.4, 4.6, 27.0, 11.0, "控制台 :8080/console",
        ["实时状态 · 决策流水 · 意识流", "情绪因果 · 人格变化 · 目标", "和她说话 / 喂食 / 睡觉 / 做梦",
         "evolve 提案列表 · QQ 调试接口"], "data", fs=7.2)
    box(ax, 60.7, 4.6, 27.0, 11.0, "外部服务",
        ["NapCat :6099（WebUI）", "Whisper :8091（听觉，可用）", "DeepSeek-VL2 :8092（视觉，停用）",
         "LoRA :8093（她自己的模型）"], "data", fs=7.2)
    box(ax, 89.0, 4.6, 27.0, 11.0, "仓库与备份",
        ["git：可回滚的代码史", "data/stickers · data/evolution/", "memory.pre-*.sqlite 备份", "models/ 本地模型"], "data", fs=7.2)

    # ================= arrows =================
    c_ext, c_io, c_perc = P["ext"][1], P["io"][1], P["perc"][1]
    c_ws, c_cog, c_llm, c_evo = P["ws"][1], P["cog"][1], P["llm"][1], P["evo"][1]
    arrow(ax, (17.2, 68.0), (17.2, 66.2), c_ext, 1.4)
    arrow(ax, (17.2, 59.0), (17.2, 57.2), c_ext, 1.4)
    arrow(ax, (17.2, 49.0), (17.2, 47.7), c_io, 1.4)
    arrow(ax, (31.5, 55.5), (36.5, 61.0), c_io, 1.8, label="消息 / 事件")
    arrow(ax, (31.5, 44.0), (36.5, 45.0), c_perc, 1.4, label="感知")
    arrow(ax, (48.1, 61.0), (48.1, 58.9), c_ws, 1.6, label="")
    arrow(ax, (59.7, 64.8), (61.3, 64.8), c_ws, 1.4)
    arrow(ax, (84.5, 66.0), (90.0, 71.0), c_llm, 1.8, label="prompt")
    arrow(ax, (90.0, 63.0), (84.5, 47.0), c_llm, 1.4, rad=-0.18, label="回复")
    arrow(ax, (90.0, 45.0), (84.5, 34.5), c_evo, 1.3, rad=0.15, label="EVOLVE")
    arrow(ax, (103.5, 51.5), (103.5, 49.9), c_llm, 1.2, label="")
    arrow(ax, (103.5, 40.5), (103.5, 39.1), c_evo, 1.2)
    arrow(ax, (31.5, 34.5), (103.5, 30.0), c_io, 1.0, rad=-0.05, ls="--", label=None)
    ax.text(66, 29.0, "输出：分条 + 打字节奏 → NapCat 发送（或她选择沉默）", fontsize=7.6, color=c_io,
            ha="center", va="center")
    arrow(ax, (60.0, 31.5), (60.0, 26.2), c_evo, 1.3, ls="--")

    ax.text(2, 1.2, "读法：① 一条消息从 QQ 进来，经感知层变成她能理解的样子；② 工作区竞争 + prompt 组装决定她此刻是什么状态；"
                    "③ LLM 生成回复或触发工具；④ 天级的自我改造闭环把「发现问题 → 评估 → 改自己 → 回看」闭上环。",
            fontsize=7.8, color=GREY, ha="left", va="bottom")

    fig.savefig(os.path.join(OUT, "architecture-structure.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
def runtime():
    fig, ax = plt.subplots(figsize=(17.6, 10.2), dpi=150)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 70)
    ax.axis("off")
    ax.text(2, 69.6, "乃爱（Noa）· 三条时间尺度", fontsize=19, fontweight="bold", color=INK)
    ax.text(2, 67.6, "一条消息（秒级）· 心跳（10 秒）· 自我改造（天级）", fontsize=9, color=GREY)

    # A: message lifecycle
    panel(ax, 1.5, 42.0, 117.0, 23.0, "A. 一条消息的生命周期（异步、可并发、她可以合法地不回）", "ws")
    xs = [3.5, 18.5, 33.5, 48.5, 63.5, 78.5]
    names = ["QQ / NapCat", "app + qq.py", "GlobalWorkspace", "状态 / 记忆", "LLM", "输出 / 落地"]
    for x, n in zip(xs, names):
        ax.text(x + 6.5, 61.8, n, fontsize=8.4, fontweight="bold", color=P["ws"][1], ha="center")

    steps = [
        (3.5, 53.0, "消息事件", ["段数组", "图片/语音/转发"]),
        (18.5, 53.0, "去重 · 限流", ["1s/人", "500 条 LRU"]),
        (18.5, 44.0, "qq.perceive", ["OCR · 语音转写", "引用/转发/卡片"]),
        (33.5, 53.0, "hear()", ["睡? 静音? 离开?", "硬上限? 插话?"]),
        (33.5, 44.0, "处理 & 评估", ["感知→情绪→预测误差", "工作区竞争取 2"]),
        (48.5, 53.0, "召回 + 记忆", ["私聊/群/公开", "分区检索"]),
        (48.5, 44.0, "拼 prompt", ["人格/自传/身体", "关系/群语料/能力"]),
        (63.5, 53.0, "生成", ["Thought/Response", "可输出（不回）"]),
        (63.5, 44.0, "工具", ["RECALL/MEMORIZE", "SEARCH/EVOLVE"]),
        (78.5, 53.0, "清洗 · 分条", ["打字节奏", "表情段"]),
        (78.5, 44.0, "落库", ["episode+interaction", "state 快照"]),
        (93.5, 53.0, "发送 / 沉默", ["NapCat", "或什么都不说"]),
    ]
    pos = {}
    for x, y, t, lines in steps:
        box(ax, x, y, 13.0, 7.2, t, lines, "ws", fill="#eff6ff", fs=7.0, tfs=8.0)
        pos[t] = (x + 6.5, y + 3.6)

    chain = ["消息事件", "去重 · 限流", "qq.perceive", "hear()", "处理 & 评估", "拼 prompt",
             "生成", "工具", "清洗 · 分条", "落库", "发送 / 沉默"]
    for a, b in zip(chain, chain[1:]):
        p1, p2 = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, lw=1.2,
                                     color="#64748b", connectionstyle="arc3,rad=0.0", zorder=2,
                                     shrinkA=10, shrinkB=10))
    ax.text(108.0, 60.5, "要点：\n· 每条消息一个 asyncio task\n· 生成放线程，不堵循环\n"
                         "· 允许不回；睡眠/静音/离开\n  都有明确代码路径", fontsize=7.4, color=GREY, va="top")

    # B: heartbeat
    panel(ax, 1.5, 4.0, 55.0, 28.0, "B. 心跳 tick()：每 10 秒一拍，没人说话时也在活着", "cog")
    items = [
        ("10s", "代谢 · 需求 · 情绪衰减 · 焦虑半衰"),
        ("10s", "超时预测（该回没回）→ 社交需求↑"),
        ("10s", "自我模型 vs 现实 → 惊讶"),
        ("每拍", "无输入也能跑工作区竞争（意识流）"),
        ("~700s", "内心独白（LLM，约 120 次/天）"),
        ("3h", "自主行动：找人 / 搜索 / 插话"),
        ("300s", "状态快照落盘（跨重启延续）"),
        ("1h+积压20", "记忆整理：蒸馏事实 / 自传"),
        ("6h", "自检：看表现，必要时自诊断"),
    ]
    for i, (w, what) in enumerate(items):
        y = 27.4 - i * 2.6
        ax.text(4.0, y, w, fontsize=7.6, fontweight="bold", color=P["cog"][1], va="center")
        ax.text(11.0, y, what, fontsize=7.6, color="#374151", va="center")
    ax.text(4.0, 5.0, "睡/醒：精力<10 或夜里<50 睡；≥80 且白天醒；睡着做梦整理；醒来写「我失去意识了 X」",
            fontsize=7.2, color=GREY, va="center")

    # C: self-improvement
    panel(ax, 58.0, 4.0, 60.5, 28.0, "C. 自我改造：天级闭环（发现问题 → 评估 → 改 → 回看）", "evo")
    steps2 = [("自诊断", ["校准 / 承诺 / 反刍", "搜索失败 / 被纠正"]),
              ("适应度", ["16 场景 × 多 seed", "子进程跑候选代码"]),
              ("DSH 提案", ["副本里改代码", "只回传 diff"]),
              ("审核 + 生效", ["apply → 重启", "7 天后回读指标"])]
    xs2 = [59.5, 74.0, 88.5, 103.0]
    for (t, lines), x in zip(steps2, xs2):
        box(ax, x, 18.0, 13.4, 10.0, t, lines, "evo", fill="#fffbeb", fs=7.0, tfs=8.2)
    for i in range(3):
        arrow(ax, (xs2[i] + 13.4, 23.0), (xs2[i + 1], 23.0), P["evo"][1], 1.5)
    ax.text(59.5, 13.4, "记忆化：self_improvement 表记录每次尝试的成败与 value_before/value_after——"
                       "试过没用的不重复试，改过的要回看", fontsize=7.2, color=GREY, va="center")

    # D: processes
    panel(ax, 1.5, -11.0, 117.0, 11.6, "D. 进程与服务（全部本机）", "data")
    procs = [("app.py :8080", "uvicorn + tick + 控制台"), ("NapCat :6099", "→ :8080 反向 WS"),
             ("hear :8091", "Whisper 可用"), ("see :8092", "VL2 停用(xformers)"),
             ("LoRA :8093", "她自己的模型"), ("DSH ACP", "按需子进程"), ("memory.sqlite", "一个文件 = 一生")]
    for i, (t, d) in enumerate(procs):
        x = 2.6 + i * 16.4
        box(ax, x, -9.4, 15.4, 8.2, t, [d], "data", fs=7.0, tfs=8.0)

    fig.savefig(os.path.join(OUT, "architecture-runtime.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    structure()
    runtime()
    print("written to", OUT)






