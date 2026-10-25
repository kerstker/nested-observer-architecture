# 乃爱（Noa）· 架构全貌

> 快照时间：2026-10-25 09:40（America/La_Paz）｜代码：`thefirstlife-dev/`｜当前 HEAD：见 `git log -1`
> 配套图：[`docs/architecture-structure.png`](architecture-structure.png)（结构）· [`docs/architecture-runtime.png`](architecture-runtime.png)（三条时间尺度）
> 重新生成：`python scripts/make_architecture_diagrams.py`

---

## 0. 一句话

**一个单进程 Python 程序**（`app.py`）：一个 10 秒一拍的状态机 + 一个精心拼装的提示词 + 一个 SQLite 记忆库，
外挂三个"小脑"（NapCat 管 QQ、Whisper 管听觉、DeepSeek 管语言），
再挂一条**天级的自我改造闭环**（自诊断 → 行为适应度评估 → DSH 在副本里改代码 → 审核生效 → 回看指标）。

规模：**约 13,000 行 Python**（含测试与脚本），核心引擎 `global_workspace.py` **2,479 行**；
**21 个测试文件 / 165 个测试**，15 个行为评估场景；**29 张表**，一个 `memory.sqlite` 就是她的全部经历。

---

## 1. 全貌：五层

```
① 外部接入       QQ 用户/群 ──► NapCat 4.18.41（OneBot v11 反向 WS）──► napcat/ 适配层
② 感知 L1        痛/触摸/喂食 · 昼夜天气 · 群语料 · 表情库 · 图片OCR · 语音 · 转发/引用/卡片
③ 内部世界       L10 GlobalWorkspace 编排：工作区竞争 + 注意力图式 + 元认知 + prompt 组装
                 身体（L3 代谢 / L4 情绪 / L6 稳态）· 价值（L5）· 记忆（L2）· ToM（L7）
                 驱动力与目标（L8）· 叙事与梦（L9）· 意识流/存在焦虑/人格稳态（L14-16）
④ 语言与工具     LLM（DeepSeek ⇄ 她自己的 LoRA 模型）· 输出协议 · RECALL/MEMORIZE/SEARCH
⑤ 自我改造       diagnosis → fitness（15 场景 × 多 seed）→ DSH 提案（ACP）→ 审核 → apply → 回看
```

数据与运维横跨所有层：`memory.sqlite`（29 表）、`/console`、`data/stickers`、`data/evolution/`、`models/`、git、备份。

---

## 2. 三条时间尺度

| 尺度 | 谁在动 | 关键周期 |
|---|---|---|
| **一条消息（秒级）** | `app.py` → `napcat/qq.py` → `GlobalWorkspace.hear()` → LLM → 分条发送 | 每条消息一个 asyncio task；生成放线程；允许沉默 |
| **心跳（10 秒）** | `proactive_sender_loop()` → `tick()` | 代谢/需求/情绪衰减、超时预测、自我模型、睡/醒、工作区竞争、独白（~700s）、自主行动（3h）、快照（300s）、记忆整理（1h 且积压≥20）、自检（6h） |
| **自我改造（天级）** | `_maybe_improve_self()` + `diagnosis` + `fitness` + `agent_bridge` | 每日 1 次额度；提案需审核；apply 后重启生效；7 天后回读指标 |

---

## 3. 一次消息的完整路径

1. **NapCat** 推事件到 `ws://127.0.0.1:8080/onebot/v11/ws/noa`；`napcat_server` 每条消息一个 task，写 `robots[last_message]`。
2. **去重限流**：500 条 message_id LRU；对"叫她"的消息按人 1 秒限流（听的不用限流）。
3. **感知翻译**：`qq.perceive(segments)` 把消息段变成"人看到的样子"——
   引用→查原消息；合并转发→展开前 8 条；卡片→标题；语音→Whisper 转写（`look=True` 时）；
   图片→**QQ 自带 OCR 读文字**，`look=True` 时再让视觉模型描述；文件/视频/骰子/位置/音乐/联系人都有对应文本。
4. **`hear()` 先做"要不要管"的判断**：睡眠？静音？已离开？硬上限？是否插话？群消息没 @ 她时可能只听不说。
5. **`process_input()`**：感官刺激 → 价值冲突 → ToM 情绪识别 → 情绪评估（记录原因）→ 预测误差 → 工作区竞争 → 记忆召回 → 生成参数。
6. **工作区竞争**：候选打分（显著度 + 噪声），阈值 0.3，容量 2；用户消息几乎总占一槽，第二槽在需求/联想/念头之间争。
7. **Prompt 组装**：人格 + 自传 + 连续性 + "此刻心里在想" + 身体心情 + 记忆分区召回 + "我记得的大家的事" + 群语料 + 表情 + **能力边界（真实）** + 关系/ToM + 环境 + 工具 + 要求。
8. **生成**：`Thought/Response`；`（不回）` 是合法输出；`Action:` 触发工具（RECALL/MEMORIZE/SEARCH/EVOLVE）。
9. **落地**：清洗（去动作描写/去工具语法）、分条、计算打字延迟（累时更慢）、表情段渲染成 OneBot segment。
10. **记忆**：`interactions` + `episodes`（带 visibility）+ 状态快照 + 因果账本；必要时触发事实抽取。
11. **输出**：`app.py` 按计划逐条 `await asyncio.sleep(delay)` 后发送；主动消息走 `output_queue`。
12. **决策留痕**：`decisions` 记录"为什么回/不回/睡着/离开/限流"，控制台可见——这是这套系统最好的可观测性设计。

---

## 4. 内部世界的几个关键机制

**工作区与"意识"**：`Workspace` 每拍竞争，胜者广播并可能成为情景记忆；`AttentionSchema` 是她对自己注意力的预测模型（注意力被意外捕获 → 惊讶）；`Metacognition` 管置信度与校准（记错/说错会被下一次对话纠正并影响她说话留余地）。

**情绪不是装饰**：六维情绪随时间半衰；每次变化都写 `state_causes`（如"预测误差：X 回得比我想的快"）；价值评估（7 个价值：尊严/真诚/亲近/好奇/自主/安稳/公平）决定冲突、生气与拒绝；价值本身也会随经历缓慢演化，并有回弹约束。

**记忆是分区的**（这一版刚改）：`episodes.visibility = private / group / public`。私聊内容不会被第三个人召回；群内共享；蒸馏出的事实（`semantic_facts`）可公开引用。整理不再只在睡眠时发生——积压 ≥20 条且间隔 1 小时就整理，所以 `semantic_facts` 从 0 长到了 36 条。

**"像不像人"的一系列约束**：说走就走（30 分钟不看那段对话）、被要求闭嘴 3 小时、被人当机器人会识别并限流、车轱辘话会腻、群发言小时上限、疲惫变慢变短、睡着只吐第一行——这些都在代码里，不是 prompt 求来的。

**性格稳态**：`trait drift` 每晚最多 ±0.05、夹在 0.1~0.9，每个变化带证据 id（`trait_changes` 已 31 条）；另有"气质 homeostasis"防止她一夜被改写。

---

## 5. 记忆与数据

`memory.sqlite` 29 张表（当前行数）：

| 类别 | 表（行数） |
|---|---|
| 对话 | `interactions`(1111) · `episodes`(880) · `inner_speech_archive`(288，遗留) |
| 事实 | `semantic_facts`(36) · `memories`(361) · `knowledge`(2) · `beliefs`(4) |
| 关系 | `users`(26) · `user_models`(19) · `relationships`(5) · `group_relations`(25) · `relation_events`(179) · `group_members`(28) |
| 自我 | `autobiography`(8) · `persona_traits`(4) · `trait_changes`(31) · `value_weights`(2) · `value_changes`(2) · `value_events`(1) · `self_state`(350) · `state_causes`(155) · `state_events`(3) |
| 行为 | `q_values`(2) · `goals`(1) · `stickers`(61) · `group_phrases`(4706) · `memes`(16) |
| 进化 | `evolution_proposals`(0) · `self_improvement`(0) |

**关于 embedding 的真相**：生产上用的是**字符 1-3gram 哈希投影（1024 维）**，不是语义向量。所以"联想回忆/切题判断/相似度去重"实际上只对字面重叠有效；同义改写不相似。要真正的语义检索，需要换 sentence-transformers 或调用 embedding API（`Embedder` 支持传 `model_name`，但生产没传）。

**膨胀**：每条 episode 带 4KB embedding；880 条 ≈ 3.5MB，全库约 4MB。检索是 O(N) 全表加载逐条点积；没有 WAL / retention / VACUUM。以目前规模没问题，到 1 万条时需要换方案。

---

## 6. 语言与工具层

- **可替换引擎**：`LLM_BACKEND = deepseek`（当前 `deepseek-flash`，远程）或 `hybrid`（`HybridEngine`：说话用她自己的 LoRA 模型，需要 JSON 的整理工作交给远程）或 `local`（llama.cpp）。全项目 prompt 是 ChatML 字符串，DeepSeek 侧用正则转 messages，并关闭思考模式。
- **输出协议**：`Thought: / Response:`；静默标记；`Action:` 语法。工具结果会拼回 prompt 让模型再回答一次。
- **搜索**：ddgs → 百度 → DuckDuckGo HTML，三级回退；全失败就明说"没查到"，不编造（有专门测试守着）。
- **知识库**：标题/正文/标签 + LIKE 检索（无向量、无去重）。
- **代码解释器**：保持关闭（exec 不是沙箱）。

---

## 7. QQ 能力层（napcat/）

| 能力 | 落点 | 状态 |
|---|---|---|
| 文本 / @ / 引用 / 表情 | `message_utils` + `qq.perceive` | 可用 |
| 图片文字（OCR） | NapCat `ocr_image` | **已验证**（真实截图可读） |
| 语音转文字 | `get_record` → Whisper :8091 | 服务可用 |
| 图片内容描述 | VL2 :8092 | **停用**（xformers 在 5090 无 kernel） |
| 合并转发 / 卡片 / 文件 / 视频 / 骰子 / 位置 / 音乐 | `perceive` | 配置已开（`parseMultMsg`） |
| 戳一戳 / 撤回 / 进群退群 / 禁言 / 管理员 / 精华 / 表情回应 / 正在输入 | `qq.notice` → 她"经历"到 | 已接 |
| 主动手势：戳回去 / 表情回应 / 撤回 / 正在输入 / 标记已读 | `qq.poke/react/recall/...` | 已接 |
| 加好友/入群请求 | 只报告，不自动同意 | 已接 |
| 群历史补读 / 群成员资料 | `get_group_msg_history` / `get_group_member_info` | 可用 |

NapCat 侧当前配置：反向 WS + 同一 token；`parseMultMsg=true`、`enableLocalFile2Url=true`、Desktop `autoStart=true`（重启会自动拉起机器人）。

---

## 8. 自我改造闭环（Level 17）

```
自诊断 diagnosis（校准/承诺/反刍/搜索失败）
   → 适应度 fitness（15 场景 × 多 seed，候选代码在子进程跑，不导入线上）
   → DSH 提案（ACP 子进程，一次性副本里改代码，只带回 git diff + 测试结果）
   → 审核（scripts/evolve.py apply 或控制台）→ 重启生效
   → 7 天后回读真实指标（value_before / value_after 写进 self_improvement）
```

- 现状：`EVOLVE_ENABLED=true`，创作者私聊 / 控制台可用；群里**根本不注入**这个工具。
- 已真实验证过一次：DSH 在副本里给 `emotion.py` 加了一行注释级改动，回传 1 行 diff，桥接跑完 165 个测试全绿，线上代码未动。
- 触发不依赖模型自觉：本地识别"改自己"的意图 → 直接进通道；也有 `!evolve` 命令与 6 小时自检。
- 仍需人插手的两步：**审核** 与 **重启**（她不能重启自己）。

---

## 9. 进程与服务

| 进程 | 端口 | 说明 |
|---|---|---|
| `app.py`（uvicorn + tick + 控制台） | 8080 | 唯一常驻进程；`/console` 与 `/onebot/v11/ws/{robot_id}` |
| NapCat Desktop（QQ NT 1000000002） | 6099（WebUI） | 反向 WS 连到 8080；OCR/取图/语音转码由它提供 |
| `scripts/senses/hear_server.py`（Whisper large-v3-turbo） | 8091 | 听觉，可用 |
| `scripts/senses/see_server.py`（DeepSeek-VL2-Tiny） | 8092 | 视觉，停用 |
| `scripts/lora/serve.py`（她自己的模型） | 8093 | 可选；`HybridEngine` 用 |
| DSH（ACP stdio 子进程） | — | 只在她发起 EVOLVE 时启动 |
| 前端轮询 `/console/api/state` | — | 每 2 秒 |

---

## 10. 控制台

`http://127.0.0.1:8080/console`（未设 `CONSOLE_TOKEN` 时仅本机）：
身体/心情/精力、6 情绪条、4 需求条、人格与变更史、决策流水、意识流、待竞争候选、内心念头、情绪因果、
"对每个人的了解 + Q 值"、最近经历、自传/自我模型、目标、注意力图式、元认知校准；
可和她说话、喂食、摸摸头、让她睡/醒/做梦/思考/心跳；
`/console/api/evolve`（提案列表/提交/应用）与 `/console/api/qq`（手工调用 QQ 动作，用于验证）。

---

## 11. 当前真实能力（别信文档，信这里）

| 能力 | 状态 |
|---|---|
| 对话、情绪、记忆分区、群聊分寸、主动行为 | 可用 |
| 图片文字 OCR、语音转写、合并转发/引用/卡片解析 | 可用 |
| 看懂非文字图片 | **不可用**（VL2 与 xformers/Blackwell 不兼容，已停用） |
| 语义级记忆检索 | **弱**（哈希 n-gram，字面匹配） |
| 自传 / 语义事实 / 人格漂移 | 可用（本轮修好后开始产出数据） |
| Q 学习 / 长期目标 | 已接，数据很少（`q_values` 2 行、`goals` 1 条） |
| 自我改造 | 通路可用，需人工审核 + 重启 |
| 用自己的 LoRA 模型说话 | 代码就绪，当前后端仍走 DeepSeek |

---

## 12. 已知缺口（按优先级）

**P0（会伤人或伤她）**
1. 测试曾经默认写生产库——已修（`test_phase0` / `test_state_linkage` 用临时库），但**新增测试要记得传 `db_path`**。
2. 私聊摘要曾可能进群（`_summarize_conversation` + `user_id OR group_id` 的旧 SQL）——需要复查现在的分区是否覆盖这条路径。

**P1（长期运行会坏）**
3. 无 WAL / `busy_timeout`；写入来自事件循环 + 写队列线程 + 控制台线程，锁冲突会被 `except` 吞掉。
4. 无 retention / VACUUM；`group_phrases` 已 4706 行（其中不少是 n-gram 碎片）。
5. 主日志（`app.stderr.log`）中文因 GBK/UTF-8 不一致乱码；无日志轮转。
6. 无进程守护；她挂掉不会自愈；重启要人工。
7. NapCat 重启依赖 Desktop 的 autoStart（已开）但登录态仍可能需要人工确认。

**P2（能力边界）**
8. embedding 不是语义向量。
9. 视觉模型不可用；OCR 只处理字，梗图含义读不懂。
10. 自我改造仍缺"她记住自己改过什么"这一段（提案结果已进 `evolution_proposals`，但没进 prompt 的长期叙事）。

---

## 13. 文件地图

| 位置 | 行数 | 内容 |
|---|---|---|
| `app.py` | 781 | 入口：配置、引擎、GW 装配、消息处理器、心跳循环、关机 |
| `napcat/` | 1,462 | `napcat_server` WS/事件；`qq.py` 感知/动作/notice；`senses.py` 视听；`message_utils`；`runtime`；`security` |
| `console/` | 251 | 控制台 API 与状态快照 |
| `minimal-self/src/core/` | 279 | clock（可模拟）、event_bus、workspace、self_state |
| `minimal-self/src/levels/l1…l10` | 4,919 | 感知 / 记忆 / 代谢 / 情绪 / 价值 / 稳态 / ToM / 驱动+目标 / 叙事 / 意识 |
| `minimal-self/src/levels/l17…l19` | 995 | 自我改造（bridge/diagnosis/fitness/解释器）、知识库、搜索 |
| `minimal-self/src/evaluation/` | 736 | 15 个行为场景 + `ScriptedLLM` / `Life` 仿真器 |
| `minimal-self/tests/` | 2,323 | 21 个测试文件，165 个测试 |
| `scripts/` | 959 | 控制台 demo、评估、senses/lora 服务、图表生成、`evolve.py` |
| `docs/` | — | 本文件 + 两张架构图 |

---

## 14. 怎么跑

```powershell
cd thefirstlife-dev
python app.py                                  # 她（需要 DEEPSEEK_API_KEY 或注册表里的 key）
python scripts\evolve.py list                  # 看自我改造提案
powershell -File scripts\senses\start_senses.ps1   # 拉起听觉(和视觉，若可用)

cd minimal-self
$env:PYTHONPATH=(Get-Location).Path
python -m unittest discover -s tests -v        # 165 个测试（也可逐文件跑）
python -m unittest tests.test_evaluation -v    # 15 个行为场景（慢，约 1 分钟）
```
