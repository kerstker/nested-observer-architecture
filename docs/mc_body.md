# 身体循环 `mc_body.PlayerBody`

她玩游戏时的身体：一个 10–20 Hz 的后台循环，替她按住方向键、像人一样转视角、一个指令做到底。
`mc_skills.Skills` 是"一次调用做一整件事"（走到树、砍掉、回来）；`PlayerBody` 是那件事真正发生时的
手和眼睛——键是持续按着的，镜头是一帧一帧转的，挖方块时眼睛一直盯着它。两者互补，不互相替代。

模块只通过 mcpfabric bridge 的 `rpc(method, params) -> dict` 与世界交互，不依赖 `Skills`，可以用假世界
（`tests/test_mc_body_loop.py` 的 `FakeWorld`）完整测试。

## 用法

```python
from src.levels.l8_agency.mc_body import PlayerBody

body = PlayerBody(game.rpc, on_event=game._log_body_event, skills_busy=game.skills.busy)

# 长目标：走一段路（阻塞调用线程，到不了就 False，不会抛异常）
body.walk_to(120, 64, -30, reach=2.0, sprint=None, timeout=25.0)

# 连续动作：走进 4.5 格、选对工具、把镜头转过去、按住左键直到方块变成空气
body.mine_block(121, 64, -30)

# 轻量航点队列：Lead 可以把 A* 或知识库给的路径喂进来
body.queue_waypoints([(120, 64, -28), (120, 64, -24)])
body.follow_path(reach=1.5)
```

所有指令方法都是阻塞的、线程安全的：从任意线程调用都会排进队列，由循环线程逐帧执行一条。指令结果
返回 `bool`；`body.stop()` 松开所有键并结束循环，`body.cancel()` 只丢掉当前指令、循环继续感知。

构造参数：

| 参数 | 含义 |
| --- | --- |
| `rpc` | `rpc(method, params) -> dict \| None`，和 `Skills` 用的是同一个 bridge |
| `on_event` | 每帧回调一次，参数是下面的状态 dict（在循环线程里执行，保持短小；不要在回调里发指令） |
| `hz` | 帧率，默认 15，限制在 10–20 |
| `skills_busy` | 可选 `callable`（例如 `skills.busy`）：它为真且没有新指令时，身体站住不动，不跟别的线程抢键盘 |
| `now` / `sleep` / `perf` | 仅在测试里替换成虚拟时钟，让整个循环瞬间跑完 |

## 公开 API

```python
PlayerBody(rpc, on_event=None, hz=15.0, *, now=None, sleep=None, perf=None, skills_busy=None)

start() -> PlayerBody           # 启动循环（发指令时也会自动启动）
stop(timeout=3.0) -> None       # 丢掉当前指令、松开所有键、结束循环；可重复调用
cancel() -> None                # 丢掉当前指令但循环继续
tick() -> dict                  # 手动走一帧（仅循环未运行时；给测试/工具用）
stats() -> dict                 # tick_rate / last action / RPC avg / p95 ...
status() -> str                 # 一句中文状态，可直接作为工具结果
state (property) -> dict        # 最近一次 player.getState

look_at(x, y, z, tolerance=2.0, timeout=2.5) -> bool
walk_to(x, y, z, reach=2.0, sprint=None, timeout=25.0) -> bool
queue_waypoints(points) -> int                 # points: (x,y,z) 元组或 {"x","y","z"}
follow_path(points=None, reach=1.5, sprint=None, timeout=30.0) -> bool
mine_block(x, y, z, timeout=10.0) -> bool
place_block(x, y, z, item=None, timeout=8.0) -> bool
attack(target, times=1, reach=3.0) -> bool     # target: uuid 字符串或 {uuid,x,y,z}
use_item(seconds=1.0, hand="main") -> bool
jump() -> bool
```

## 每帧事件（`on_event`）

```python
{
  "event": "tick",
  "t": 1234.5678,                  # 单调时钟秒
  "tick": 812,                     # 累计帧号
  "action": "idle" | "walk" | "look" | "mine" | "place" | "attack" | "use" | "jump" | "yield",
  "busy": True,                    # 当前是否有指令在跑
  "pos": {"x": 120.5, "y": 64.0, "z": -30.5},
  "moved": 0.287,                  # 相对上一帧的位移（格）
  "yaw": 12.34, "pitch": -3.2,
  "onGround": True,
  "health": 20.0, "food": 20,
  "sprinting": False,
  "target": [121, 64, -30] | "uuid" | None,
  "rpc_ms": 4.2,                   # 本帧 player.getState 的耗时
  "tick_rate": 15.0,
  "note": "",                      # 短中文说明（例如"前面有台阶，跳一下"）
}
```

`stats()` 返回 `running/busy/action/target/note/ticks/tick_rate/rpc_calls/rpc_errors/rpc_avg_ms/rpc_p95_ms/waypoints`。

## 像人的地方

- 镜头不瞬移：`control.look` 每帧发一小步，最快约 300°/s；起步有加速（约 0.12 s 到全速），离目标
  几度时减速（settle），大角度转弯需要十几帧。
- 走路：持续 `control.setInput {forward, ...}` + 转向；前方 0.6 格周期性探测（每 ~0.12 s 一次
  `perception.blocks`）——一格台阶会跳上去，两格高的墙会先尝试绕、约 2.6 s 后放弃；前方是两格以上
  的坑则不走过去（除非目标本身在下面）。长直平地自动疾跑。
- 挖方块：先 `perception.blocks {tool:true}` 看 `bestSlot/canHarvest`，走到 4.5 格内，用视线检测确认
  没有别的方块挡着，转过去，`interact.breakBlock {mode:"survival"}` 后每 0.2 s 轮询直到变成空气。
- 放方块：`interact.placeBlock {x,y,z,face}` 的 `x,y,z` 是**被点的方块**，不是要放的位置；模块自动挑
  一个相邻的实心方块（优先地板 → 墙），算出朝向它的 `face`。
- 打怪：走到 3 格内、看向目标身体、`interact.attackEntity {uuid}`，每 ~0.65 s 一次。
- 安全：持续拿不到 `player.getState` 时松键；`stop()`/`cancel()`/指令失败/超时后都保证键盘是松开的。

## 限制

- `perception.blocks` 依赖客户端的视野/加载能力；看不到的方块不会凭空去挖（返回 `False` 并写 `note`）。
- 只做局部避障（0.6 格探测 + 卡住重试），不是完整 A*：远距离或迷宫请由 Lead 用 `nav.pathTo` 之外的
  方式给出航点，或让 `Skills` 负责长程寻路，`PlayerBody` 负责最后一段和连续动作。
- 打怪不自己找目标：`uuid` / 坐标由 Lead 从感知里取。
- 一个身体同一时间只有一条指令；不要在 `on_event` 回调里发指令（会直接抛 `RuntimeError`，避免自锁）。
- 放置/挖掘的服务器确认字段（`placed`、`result == "OUT_OF_REACH"`）按 mcpfabric 当前实现处理；远程服
  务只做预测，必要时以随后的 `perception.blocks` 为准。
