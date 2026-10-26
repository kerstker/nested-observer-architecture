# minimal-self — 她的心智

这一层是 Noa 的认知内核：`src/levels/` 里一层一个文件夹，`src/core/` 是共用底座，
`src/evaluation/` 是行为场景与仿真器，`tests/` 是测试。

完整、随代码更新的架构说明在仓库根目录的 [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)；
这里只放一张速查表。

## 层级

| 层 | 目录 | 一句话 |
|---|---|---|
| L1 | `l1_perception/` | 感官、环境、群语料、表情、文件→文字 |
| L2 | `l2_memory/` | 情节 / 语义 / 画像 / 对话时间线 / 关系图 |
| L3 | `l3_metabolism/` | 精力与睡眠 |
| L4 | `l4_emotion/` | 六种情绪、压制、习惯化、心境 |
| L5 | `l5_value/` | 七个价值与冲突 |
| L6 | `l6_homeostasis/` | 饥饿/孤独/好奇/卫生 |
| L7 | `l7_tom/` | 情绪识别、潜台词、对每个人的模型、传闻 |
| L8 | `l8_agency/` | 工具、权限、防火墙、MCP、游戏身体、驱力、目标 |
| L9 | `l9_narrative/` | 日记/自传/人格漂移/说话范例 |
| L10 | `l10_consciousness/` | 全局工作空间、注意力图式、元认知 |
| L17 | `l17_self_improvement/` | 自诊断、维护请求、改造提案 |
| L18/L19 | `l18_knowledge/`, `l19_search/` | 知识库、三级回退搜索 |

## 跑测试

```bash
cd minimal-self
python -m pytest -q          # 416 个测试，约 7 分钟，不需要 API Key / 网络
```

测试是 unittest 风格，pytest 只是 runner；没有 pytest 时 `python -m unittest` 也可以。
`pytest.ini` 里的 `testpaths = tests` 是有意的：本目录下的 `data/`、`models/`、虚拟环境
不是测试，别让 pytest 去扫描它们。

## 约好的一件事

`src/levels/` 下的每一层都可以单独实例化和测试（传临时 `db_path`），
**L10 的 `global_workspace.py` 只做编排**。改任何一层之前，先看它有没有对应的测试和评估场景
（见 `src/evaluation/` 与根目录 `scripts/` 下的审计脚本）。
