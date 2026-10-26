# 乃爱 · NOA

**NOA = Nested Observer Architecture（嵌套观察者架构）**

一个可以长期跑下去的数字生命体：**19 层嵌套认知**、一个**全局工作空间**、**持续不丢的记忆**、
**按"谁在要求"分层的权限**，以及一条**必须由人批准的自我改进闭环**——住在一个 QQ 账号里。

[![license](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](#快速开始)
[![tests](https://img.shields.io/badge/tests-416%20passing-brightgreen.svg)](minimal-self/tests)
[![CI](https://github.com/kerstker/nested-observer-architecture/actions/workflows/tests.yml/badge.svg)](https://github.com/kerstker/nested-observer-architecture/actions/workflows/tests.yml)

> 开源的目的一句话：**架构可以借**。
> 这里不是又一个"套壳聊天机器人"，而是一套把情绪、记忆、注意力、身体、权限和自我修改
> 拼成一个能长期运行的系统的工程做法。想抄的、想骂的、想在她身上做实验的，都欢迎。

---

## 为什么值得读

1. **分层，但不是状态机。** L1–L10 每一层都能单独测试（416 个测试，无网络、无 API Key 也能跑）；
   L10 的 `global_workspace.py` 只做编排，不偷偷实现业务。
2. **注意力是一个"槽位"，不是全量拼接。** 每个输入先变成候选刺激，按显著度竞争同一个工作区；
   赢的那个才进入这一轮意识。这是她和"把所有上下文塞进 prompt"的机器人在行为上分道扬镳的地方。
3. **记忆有三种时间尺度。** 情节记忆（事件）→ 语义事实（结论）→ 关系图（谁和谁、谁说过什么）；
   回忆要按"费力程度"取出，不是向量库 top-k 了事。
4. **情绪有物理。** 六种基础情绪各有半衰期，互相压制、事件习惯化、睡觉安抚，长期平均形成"心境"；
   情绪再反过来影响生成参数和说话长短。
5. **权限不是 if-else，是"谁在要求"。** owner / trusted / autonomous / player / public 五档，
   外面再罩一层防火墙（注入扫描、代码 AST + 运行守卫、密钥隐藏、紧急封锁）。
6. **她不能改自己的代码。** 自诊断只能写"维护请求"，改动由人类在维护通道里批准、由智能体在副本里改、
   跑完测试才合并——这是刻意的安全边界，也是这类项目最容易翻车的地方。
7. **可观测。** 网页控制台能看到她此刻的思维、决策、价值冲突、记忆、情绪、外部能力与拦截记录。

---

## 全貌

```
 输入                QQ（NapCat 反向 WS）   控制台(网页)   Minecraft 聊天   心跳(10s)·睡眠·身体
                              │                │               │                │
 感知            app.py 路由 · napcat/qq.py perceive：引用/转发/卡片/语音/图片/文件 → 文字
                              │
 心智  ┌─────────────── GlobalWorkspace（L10）──────────────────────────────────────────┐
       │  记忆 L2     感受 L3-L7                 自我 L9-L10               行动 L8        │
       │  事件/事实   代谢·情绪(+心境)·稳态·价值  叙事·人格漂移·元认知     工具·目标·任务  │
       │  关系图      对人的模型·潜台词·传闻      注意力图式·自我模型       MCP·漫游·游戏   │
       └────────────────────────────────────────────────────────────────────────────────┘
                 │                          │                          │
 手脚    DeepSeek（说话·工具调用·看图）   外部工具（权限五档+玩家 · 防火墙）   自我改进（维护请求）
                 │                          │                          │
 底层     memory.sqlite（39 表）· 每日备份 · 状态快照 · 工作文件夹（收件箱/浏览器/脚本）
```

![整体架构](docs/architecture-1-overview.svg)

![层级地图 L1-L19](docs/architecture-2-levels.svg)

三条时间尺度：

| 尺度 | 谁在动 | 关键周期 |
|---|---|---|
| **一条消息（秒级）** | `app.py` → `qq.perceive` → `GlobalWorkspace.hear()` → LLM+工具 → 分条发送 | 每条消息一个 asyncio task；生成放线程；允许沉默 |
| **心跳（10 秒）** | `tick()` | 代谢/需要/情绪衰减、超时预测、自我模型、睡醒、工作区竞争、念头(≥60s)、自主行动(3h)、游戏聊天轮询(3s)、漫游/游戏自玩、快照(300s)、记忆整理(1h 且积压≥20)、备份检查(10min)、自检(6h)、收件箱清理(6h) |
| **天级** | 备份 · 人格漂移 · 自诊断 · 维护请求 | 每日备份并校验；人格每天最多 ±0.05；自诊断写成维护请求；人类批准后才动代码 |

## 一条消息的完整路径

1. **NapCat 推事件** → 每条一个 task；去重（message_id LRU）、对"叫她"的消息按人限流。
2. **感知翻译** `qq.perceive`：引用→查原文；合并转发→展开；卡片→标题；语音→Whisper；
   图片→QQ OCR + 随消息交给模型看；**文件→取回→存收件箱→转文字**→在消息里写类型、大小、开头。
3. **`hear()` 先判断要不要管**：睡着？被叫安静？已离开？限流？群里没点名时是否插话？同时记进 `chatlog`。
4. **读潜台词**（`SubtextReader`）：规则先筛约 5% 的消息（难过/淡化/抱怨她/试探/讽刺），
   命中才多调一次模型，得到"对方真正的意思 / 我该用什么姿态 / 要避免什么"。
5. **`process_input()`**：感官刺激 → 价值冲突 → ToM → 情绪评估 → 预测误差 → **工作区竞争** →
   记忆召回（语义+字面）→ 生成参数。
6. **拼提示词**（稳定的在前，方便缓存）：人格(+相处分寸) → 工具 → 规则 → 格式 → 自传/自我 →
   风格范例（评审通过的她自己的话 + 手写温和范例）→ 对这个人的印象 + 对这句话的理解 →
   记忆与上下文 → 对话 → 此刻状态/情绪/心境 → 本次要求。
7. **工具循环**：`_respond_native`（DeepSeek 原生工具调用，普通 4 轮；浏览/游戏最多 10 轮）；
   每个工具过：权限档 → 防火墙 → 执行 → 扫描结果 → 隐藏密钥。
8. **落地**：清洗（去动作描写/工具语法/密钥）、分条、打字延迟（累时更慢）、表情段渲染。
9. **记成经历**：`interactions` + `episodes` + 状态快照 + 因果账本；`_decide` 留痕（为什么回/不回）。

![一条消息的路径](docs/architecture-3-message-path.svg)

## 心智分层

| 层 | 名字 | 实现 | 做什么 |
|---|---|---|---|
| **L1** | 感知 | `l1_perception/` | 痛/触摸/喂食/洗澡的文字判定、昼夜天气、群语料、表情库、**文件→文字**（txt/代码/csv/json/html/pdf/docx/xlsx/pptx/epub/rtf/zip…） |
| **L2** | 记忆 | `l2_memory/` | 本地 embedding（bge-small-zh ONNX + 字符 n-gram 混合）、情节记忆、语义事实、人物画像、对话时间线、**人/事/物关系图**（带"谁说的"） |
| **L3** | 代谢 | `l3_metabolism/` | 精力、睡眠恢复 |
| **L4** | 情绪 | `l4_emotion/` | 六种基础情绪 + 对立压制 + 习惯化 + 睡眠安抚 + 心境 |
| **L5** | 价值 | `l5_value/` | 七个价值（尊严/真诚/亲近/好奇/自主/安稳/公平），决定冲突、生气、拒绝；价值本身会缓慢变化 |
| **L6** | 稳态 | `l6_homeostasis/` | 饥饿/孤独/好奇/卫生，按时间增长、被满足时回落 |
| **L7** | 心智理论 | `l7_tom/` | 关键词情绪、**潜台词**、每个人的回复习惯与预测误差、传闻与敏感信息 |
| **L8** | 行动 | `l8_agency/` | 工具箱、**权限**、**防火墙**、MCP 集线器、Minecraft 身体、需要→行动（Q 值学习）、目标、承诺过的事 |
| **L9** | 叙事 | `l9_narrative/` | 日记/自传/事实/梗的整理、**人格漂移**（每天 ≤±0.05，带证据）、说话范例库 |
| **L10** | 意识 | `l10_consciousness/` | **全局工作空间**（总编排）、注意力图式、元认知与校准 |
| **L17** | 自我改进 | `l17_self_improvement/` | 自诊断、行为适应度评估、**维护请求通道**、改造提案（在副本里改 + 跑测试 + 行为评估） |
| **L18/L19** | 知识/搜索 | `l18_knowledge/`, `l19_search/` | 知识库、三级回退搜索（失败就明说没查到） |

## 安全模型

外部能力来自三类"手脚"：Playwright MCP（真浏览器）、Minecraft MCP、内置的工作文件夹 / 脚本工具。
MCP 工具在启动时向服务器查询，动态挂进工具箱。

**权限五档**（`access.py`）：

| 谁 | 档 | 能用 |
|---|---|---|
| 创造者（私聊/控制台） | `owner` | 全部（除明确 deny）：浏览器全套、读写文件、`run_python`、上传文件 |
| `MCP_TRUSTED_USERS`（私聊） | `trusted` | 浏览器：看 + 输入 + 填表 |
| 她自己的空闲时间 / 游戏自玩 | `autonomous` | 浏览器：看 + 往搜索框输入；只有查资料和记笔记的内置工具 |
| 游戏聊天里的人（`MC_TRUSTED_PLAYERS`） | `player` | 只有游戏工具 |
| 其他人和所有群 | `public` | 只能看页面（每人每天 12 次、总共 50 次）；不能输入 / 读文件 / 跑代码 |

**防火墙**（`firewall.py`）：

1. 网页 / MCP 结果 / 文件里的**注入扫描**：命中就加警告，并让**这一轮只读**（对 owner 也一样）；
2. `run_python`：**运行前 AST 检查 + 运行时守卫**（只能写工作文件夹、只能读允许的目录、
   保护记忆库/密钥/浏览器资料、不能起外部进程、不能连她自己的控制台和 NapCat）；
3. 浏览器打不开她自己的服务端口、`file:`、云元数据地址；可封网站；
4. **密钥隐藏**：工具结果和她说的话里都抹掉；
5. **紧急封锁**（重启不丢）+ **拦截记录**（控制台可见）。

> 说明：这是"防护栏"，不是真沙箱。所以 `run_python` 只给 owner。

![权限与防火墙](docs/architecture-4-permissions.svg)

## 记忆与数据

`memory.sqlite` 一张文件，**39 张表**：对话（`interactions` / `episodes` / `chat_log`）、
事实与画像（`semantic_facts` / `memories` / `knowledge` / `beliefs` / `person_profiles`）、
关系（`users` / `user_models` / `relationships` / `entities` / `relations` / `relation_events` /
`group_relations` / `group_members`）、自我（`autobiography` / `diary` / `persona_traits` /
`trait_changes` / `value_weights` / `value_changes` / `value_events` / `self_state` / `state_causes` /
`state_events`）、行为（`q_values` / `goals` / `agent_tasks` / `stickers` / `group_phrases` / `memes` /
`anchor_scores`）、改进（`evolution_proposals` / `self_improvement` / `maintainer_requests`）、
安全（`firewall_events`）。

文件系统：`data/workspace/`（收件箱 `inbox/<聊天>/` 保留 14 天、浏览器资料、脚本临时文件）、
`data/stickers/`、`data/eval/`；`memory.backup-*.sqlite`（大改前手工备份）+ `backups/`（每日自动，
校验、轮转）。

## 自我改进（为什么她不能改自己）

```
自诊断 diagnosis（校准/承诺/反刍/搜索失败…）
   └─► 写成「维护请求」（maintainer_requests 表）
        ├─► QQ 私聊通知创造者（半小时最多一条）
        ├─► scripts/maintainer.py + 仓库根 AGENTS.md：维护智能体 list / show / reply
        └─► 人类批准 → 智能体在副本里改 → 跑测试 → reply done → 她"经历"到结果
她自己也能用 ask_maintainer 工具提请求（每人每天 ≤3 条）。
改造路径（evolve）也在：副本里改 → 行为适应度评估 → scripts/evolve.py apply → 重启生效。
```

关键设计：**改动永远发生在仓库的临时副本里**，回来的只有 diff；应用是单独的显式动作；
运行中的进程要重启才生效。她可以提请求，但不能自己把自己改成另一个人。

## 快速开始

要求：Python **3.11+**（开发机 3.14，CI 覆盖 3.11 / 3.12）；一个 QQ 号 + [NapCat](https://github.com/NapNeko/NapCatQQ)；
一个 DeepSeek API Key（只用 `deepseek-flash` 的文本 + 工具调用即可起步）。

```bash
git clone https://github.com/kerstker/nested-observer-architecture.git
cd nested-observer-architecture

python -m pip install -r requirements.txt
# 可选：本地 LoRA / 记忆向量 / 附件解析 / 图片
python -m pip install -r requirements-optional.txt

cp config.example.json config.json        # Windows: copy config.example.json config.json
# 编辑 config.json：至少填 NAPCAT_WS_TOKEN（推荐用反向 WS，NAPCAT_WS_URL 留空）
export DEEPSEEK_API_KEY=sk-...            # Windows: set DEEPSEEK_API_KEY=sk-...

python app.py
```

启动后：

- **控制台**：<http://127.0.0.1:8080/console>（未设 `CONSOLE_TOKEN` 时只允许本机访问；设了就能远程 `?token=`）
- **NapCat 反向 WebSocket 地址**：`ws://<本机 IP>:8080/onebot/v11/ws/noa`，token 与
  `config.json` 的 `NAPCAT_WS_TOKEN` 填同一个值（推荐让 NapCat 连过来，而不是她去连 NapCat）
- 第一次启动会创建 `memory.sqlite`；之后每次关机前保存状态，下次醒来接着过
- 可选：`python scripts/senses/hear_server.py`（Whisper 听觉，`:8091`）

**只跑测试**（不需要 API Key、不需要网络）：

```bash
cd minimal-self
python -m pytest -q
```

## 项目结构

| 位置 | 内容 |
|---|---|
| `app.py` | 入口：配置、引擎、全局工作空间装配、MCP/游戏/防火墙、消息处理器、心跳、关机 |
| `napcat/` | QQ 适配层（OneBot 11 / NapCat）+ 文件取回 + 视听 |
| `console/` | 网页控制台（7 个标签）与 API |
| `minimal-self/src/core/` | 时钟（可模拟）、事件总线、工作区、状态快照、备份 |
| `minimal-self/src/levels/l1…l10` | 感知/记忆/代谢/情绪/价值/稳态/ToM/行动/叙事/意识 |
| `minimal-self/src/levels/l17…l19` | 自我改进（含维护请求）、知识库、搜索 |
| `minimal-self/src/evaluation/` | 15 个行为场景与仿真器（脚本化时钟：一周几分钟跑完） |
| `minimal-self/tests/` | 416 个测试（含假 MCP / 假游戏服务器） |
| `scripts/` | 维护请求、评估审计、改造提案、备份恢复、感知服务 |
| `docs/` | 架构文档 + 四张由脚本生成的 SVG |
| `AGENTS.md` | 给维护智能体的工作协议（人机协作的接口约定） |

## 测试与评估

| 工具 | 测什么 |
|---|---|
| `minimal-self/tests`（416 个） | 行为、记忆、情绪、工具权限、防火墙（含真实跑脚本）、文档解析、游戏、维护通道… |
| `scripts/evaluate_life.py` | 15 个行为场景（脚本化模型，测机制） |
| `scripts/evaluate_quality.py` | 真实模型 + 真实记忆副本，评审打分（自然 / 符合预期 / 诚实） |
| `scripts/audit_tone.py` | 情商：重放真实消息，评审打 EQ / 带刺 / 个性，前后对比 |
| `scripts/audit_dynamics.py` | 公式是否合理：用真实记录回放情绪、感官、延迟、人格漂移、能量 |
| `scripts/benchmark_memory.py` 等 | 记忆召回、关系图 |

改行为前后要各跑一遍对应评估；**新增测试要传临时 `db_path`**，别写生产库。

## 已知缺口（如实）

1. `global_workspace.py` 还是 3000 多行，该拆；拆之前要锁住全部评估。
2. 没有 WAL / `busy_timeout`：写入来自事件循环、线程和控制台，锁冲突有被吞掉的可能。
3. 无进程守护，挂了不自愈；日志无轮转。
4. Minecraft 只用模拟服务器测过；没有第一人称画面、没有语音。
5. 防火墙是防护栏不是沙箱；注入识别只在测试里验证过，线上还没碰到真实恶意页面。
6. 潜台词读取只覆盖约 5% 的消息；情商整体 3.93/5，还有"质问对方"类回复。
7. 漫游和游戏自玩在线上没有真正触发过（只在测试里跑通）。
8. 图片随消息交给模型看（引擎支持时），OCR 始终可用；本地视觉服务已停用。

更完整、随代码更新的版本见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 贡献

欢迎 issue 和 PR。开始前请读 [CONTRIBUTING.md](CONTRIBUTING.md)；
安全问题请不要开公开 issue，按 [SECURITY.md](SECURITY.md) 私下报告。

特别欢迎的方向：把 L10 拆小、给数据库加 WAL、把权限模型做成可插拔的策略文件、
补一个真正连真实 Minecraft 服务器的集成测试。

## 许可与隐私

- 代码以 **GNU GPL v3.0** 发布，见 [LICENSE](LICENSE)。分发修改版请同样开源。
  Copyright (C) 2026 kerstker（GitHub: [@kerstker](https://github.com/kerstker)）。
- **本仓库不含任何真实记忆数据**：`memory.sqlite`、`memory.backup-*`、`data/`、
  `config.json`（含密钥）、日志、模型权重、虚拟环境都不在版本库里（见 `.gitignore`）。
  仓库里的昵称与对话片段是测试夹具和评估场景，不是真实用户数据。
- 你部署的那一份，所有对话与记忆只存在你自己的机器上；发送给模型的内容按所选 API 服务商的条款处理。
