# Minecraft 运行接线与验收

`GameBridge` 使用 `gw.episodes.embedder` 原实例创建独立的 `GameMemory`。游戏向量与语义记忆处于相同空间，但游戏检索只查询 `mc_game_memory.sqlite`，不会搜索 QQ 对话。`mc_recall` 提供显式语义检索，`_scene()` 在目标和当前位置附近自动召回。

## 事件与地图

- 工具返回“开始做”不记为成功；`Skills.on_action` 在后台任务结束后记录实际结果。
- `Places.on_change` 将家、箱子内容、矿点等变化送入向量库；相同箱子观测不重复写入。
- `Experience.on_learn` 将游戏经验送入向量库。
- `GameRuntime` 的有界队列在单独线程进行嵌入和数据库写入。队列满时丢弃并增加 `dropped`，不阻塞身体；`errors` 和 `pending` 可通过游戏状态查看。
- 地图按维度区分家和箱子，旧版无维度记录按主世界解释。游戏数据库仍按此部署存储；切换到另一服务器前需要独立存储，当前没有跨服务器地图迁移。

## 玩家控制与决策

`Skills.body` 连接到 15 Hz `PlayerBody`；短距离移动、视角、挖放由身体执行，长距离保留现有寻路。`auto_work()` 在技能空闲后调用 `ProPlanner`，执行目标级任务。战斗和吃饭可取消当前身体命令，暂停技能，避免两套控制争抢。

`set_home` 是选择落脚点并安置设施；`build_hut` 才是搭屋。搭屋状态来自实际方块，靠近家时最多每五秒复查。未完成搭屋后冷却两分钟，期间仍能执行其它目标；不能把背包中有方块当成已有墙或屋顶。

`QuestBook` 直接加载当前实例 `config/ftbquests/quests`，保留依赖链、击杀数量和物品组件限制。可用时每五秒读取桥的 `quests.progress`：数据来自 FTB 客户端 `ClientQuestFile.selfTeamData`，由服务端同步包更新，覆盖本地库存推断。它读取实际完成状态、任务计数、是否允许开始和未领奖标记，不发送完成/领奖包。

服务端快照仅保存在内存，断线立即失效，二十秒未刷新视为过期，新的不透明会话 ID 清除旧会话完成通告。无数据或 API 不兼容会明确报告不可用；此时仅保留本机观察到的物品条件/实际到访维度，不能据此断言击杀、结构或服务器任务完成。严格组件任务不会仅靠基础物品 ID 判定完成。当前 BMC5 的 143 任务可在单页 256 项内读取；超过单页的整合包需要分页聚合，不能把部分页作为完整进度。

桥采用可选反射兼容，无 FTB 硬依赖，保留已有 loopback/token 边界和 `enableVision` 门控；客户端主线程只读自己的团队，不输出团队 UUID、玩家身份或服务器地址。实际安装的 `ftb-quests-neoforge-2101.1.34.jar` 已用 `javap` 核对这些公开 API 和服务端同步写入路径。此功能不代表已经通关，任务执行仍受技能实际能力限制。

后台动作确有成功证据时，下一目标可在两秒后衔接；失败仍按普通间隔退避，每小时启动上限保留。健康状态下龙战交由末地控制器处理水晶、弓箭和龙息，普通敌人和低血逃生仍保留身体反射。

## 直播公开快照

状态写入部署根目录 `live/overlay/game_state.json`，独立线程每 100 ms 合并写入最新快照，使用临时文件替换。字段为：

```text
t: Unix 时间秒
body: action,busy,pos,yaw,pitch,moved,health,food,onGround,sprinting,sneaking,usingItem,tick_rate,dimension
doing,next,rationale,goal,home
```

这些字段只来自游戏状态和规则规划说明，不读取内部独白、QQ 消息、用户档案、令牌或原始工具日志。`rationale` 是公开的简短行动理由。读端应检查 `t`，过期时显示连接中/状态过期，不能继续展示旧动作。

## 在线验收

重载应用后，从控制台游戏状态确认：

1. `game_memory.shared_embedder=true`，`backend/model_id` 与语义记忆一致。
2. `body.tick_rate` 的实测值及 RPC 延迟；配置 15 Hz 本身不是达标证据。
3. 完成真实技能后 `game_memory.records` 增长，`memory_queue.errors/dropped` 未增长。
4. 同位置箱子更改内容后能召回新观测；换维度不会把主世界家当成本维度目标。
5. 浏览器 HUD 读到新鲜 `t`，动作来自身体状态；身体空闲、断连时不播放伪造的工作动作。

离线测试：`python -m pytest -q tests/test_game.py tests/test_mc_runtime.py tests/test_mc_game_memory.py tests/test_mc_memory.py tests/test_mc_places.py`（在 `minimal-self` 执行）。测试使用临时目录/内存库，不向在线机器人灌入测试消息。

在项目根目录执行 `python scripts/verify_mc_runtime.py --require-live`，可只读核对在线实例的共享 Embedder 标志、FTB 加载来源/数量/错误，以及当前游戏向量库的模型、维度和召回。脚本只连接本机控制台，并先把稳定的 DB+WAL 文件复制到临时目录，SQLite 不会打开生产路径，也不会更新召回计数或迁移向量。默认不输出游戏记忆正文。共享实例身份由正在运行的应用报告；脚本自身创建的 Embedder 仅用于验证向量空间，不能替代运行时身份检查。
