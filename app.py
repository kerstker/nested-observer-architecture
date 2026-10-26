import uvicorn
import time
from napcat.napcat_server import app as napcat_app
from napcat.napcat_server import register_message_handler, register_event_handler, set_access_token, is_own_message
from napcat.qq import qq
from napcat.senses import senses
qq.senses = senses
from napcat.event_models import MessageEvent
from napcat.message_utils import extract_plain_text, should_trigger, split_reply
from napcat.runtime import RuntimeManager, exception_protected
from napcat.ws_client import WSClient
import logging
import json
import os
import asyncio
from typing import Dict, Any
from collections import deque
from datetime import datetime

# 导入新创建的模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "minimal-self"))

# Configure Logging (Use the new logger governance)
# Try to import from root src first, or fallback
try:
    from src.core.logger import setup_logging
    logger = setup_logging()
except ImportError:
    # Fallback if path issues
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

from schemas import LLMOutput
from prompting import PromptContext, DEFAULT_PERSONA, DEFAULT_STATE, DEFAULT_POLICY
from inference import InferenceEngine
from policy import PolicyManager

# Import Level 1-10 Framework
from src.levels.l4_emotion.emotion import EmotionState
from src.core.event_bus import EventBus
from src.levels.l10_consciousness.global_workspace import GlobalWorkspace
from src.analysis.group_relations import RealTimeGroupRelationAnalyzer
from src.levels.l1_perception.stickers import StickerMemory

# 配置示例
"""
三种 token 注入方式的 config 写法示例：

1. Header 注入（Authorization: Bearer xxx）
{
    "NAPCAT_WS_URL": "ws://<windows-ip>:<port>/onebot/v11/ws",
    "NAPCAT_WS_TOKEN": "your-token-here",
    "NAPCAT_WS_TOKEN_HEADER": "Authorization",
    "NAPCAT_WS_TOKEN_IN_HEADER": true,
    "NAPCAT_WS_TOKEN_IN_QUERY": false
}

2. Header 注入（自定义 header 名称）
{
    "NAPCAT_WS_URL": "ws://<windows-ip>:<port>/onebot/v11/ws",
    "NAPCAT_WS_TOKEN": "your-token-here",
    "NAPCAT_WS_TOKEN_HEADER": "access_token",
    "NAPCAT_WS_TOKEN_IN_HEADER": true,
    "NAPCAT_WS_TOKEN_IN_QUERY": false
}

3. URL Query 注入
{
    "NAPCAT_WS_URL": "ws://<windows-ip>:<port>/onebot/v11/ws",
    "NAPCAT_WS_TOKEN": "your-token-here",
    "NAPCAT_WS_TOKEN_QUERY_PARAM": "access_token",
    "NAPCAT_WS_TOKEN_IN_HEADER": false,
    "NAPCAT_WS_TOKEN_IN_QUERY": true
}
"""

# 读取配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# NapCat 连过来时必须带上这个 token（在 NapCat 的网络配置里填同一个值）
_napcat_token = config.get('NAPCAT_WS_TOKEN', '')
set_access_token('' if _napcat_token == 'your-token-here' else _napcat_token)

# 创建 WebSocket 客户端
ws_client = WSClient(
    ws_url=config.get('NAPCAT_WS_URL', 'ws://localhost:3000/onebot/v11/ws'),
    token=config.get('NAPCAT_WS_TOKEN', ''),
    token_header=config.get('NAPCAT_WS_TOKEN_HEADER', 'Authorization'),
    token_query_param=config.get('NAPCAT_WS_TOKEN_QUERY_PARAM', 'access_token'),
    token_in_header=config.get('NAPCAT_WS_TOKEN_IN_HEADER', True),
    token_in_query=config.get('NAPCAT_WS_TOKEN_IN_QUERY', False),
    heartbeat_interval=config.get('NAPCAT_WS_HEARTBEAT_INTERVAL', 30),
    max_reconnect_time=config.get('NAPCAT_WS_MAX_RECONNECT_TIME', 30)
)

# OneBot Actions 类（简化版，使用 WebSocket 发送）
class OneBotActions:
    """OneBot Actions 类，使用 WebSocket 发送请求"""
    
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        self.echo_id = 0
        try:
            from napcat.onebot_client import get_or_create_client
            self._http_client_factory = get_or_create_client
        except Exception:
            self._http_client_factory = None
    
    def _get_echo(self) -> str:
        """获取唯一 echo ID"""
        self.echo_id += 1
        return f"action_{self.echo_id}_{int(time.time() * 1000)}"
    
    async def send_private_msg(self, user_id: int, message: Any) -> bool:
        """发送私聊消息
        
        Args:
            user_id: 用户 ID
            message: 消息内容
            
        Returns:
            bool: 是否发送成功
        """
        from napcat.napcat_server import ws_manager
        
        action_data = {
            "action": "send_private_msg",
            "params": {
                "user_id": user_id,
                "message": message
            },
            "echo": self._get_echo()
        }
        
        try:
            ws_ok = await ws_manager.send_to_robot(self.robot_id, action_data)
        except Exception as e:
            logger.error(f"Failed to send private message via WS: {e}")
            ws_ok = False
        if ws_ok:
            return True
        
        # 若未连接 WebSocket，则回退到 HTTP API
        base_url = config.get('NAPCAT_API_BASE')
        token = config.get('NAPCAT_API_TOKEN')
        if self._http_client_factory and base_url:
            try:
                from napcat.onebot_client import get_or_create_client
                client = get_or_create_client(self.robot_id, base_url, token)
                await client.send_private_msg(user_id, message)
                return True
            except Exception as e:
                logger.error(f"HTTP fallback failed (private): {e}")
                return False
        else:
            logger.error("No WS connection and HTTP API base not configured for private message.")
            return False
    
    async def send_group_msg(self, group_id: int, message: Any) -> bool:
        """发送群消息
        
        Args:
            group_id: 群 ID
            message: 消息内容
            
        Returns:
            bool: 是否发送成功
        """
        from napcat.napcat_server import ws_manager
        
        action_data = {
            "action": "send_group_msg",
            "params": {
                "group_id": group_id,
                "message": message
            },
            "echo": self._get_echo()
        }
        
        try:
            ws_ok = await ws_manager.send_to_robot(self.robot_id, action_data)
        except Exception as e:
            logger.error(f"Failed to send group message via WS: {e}")
            ws_ok = False
        if ws_ok:
            return True
        
        # 若未连接 WebSocket，则回退到 HTTP API
        base_url = config.get('NAPCAT_API_BASE')
        token = config.get('NAPCAT_API_TOKEN')
        if self._http_client_factory and base_url:
            try:
                from napcat.onebot_client import get_or_create_client
                client = get_or_create_client(self.robot_id, base_url, token)
                await client.send_group_msg(group_id, message)
                return True
            except Exception as e:
                logger.error(f"HTTP fallback failed (group): {e}")
                return False
        else:
            logger.error("No WS connection and HTTP API base not configured for group message.")
            return False

# 创建 OneBot Actions 实例
actions = OneBotActions("test_robot")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置内部日志
INTERNAL_LOG_FILE = "data/noa_internal.log"
# 创建数据目录
os.makedirs("data", exist_ok=True)
# 配置内部日志记录器
internal_logger = logging.getLogger("noa_internal")
internal_handler = logging.FileHandler(INTERNAL_LOG_FILE, encoding="utf-8")
internal_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
internal_logger.addHandler(internal_handler)
internal_logger.setLevel(logging.INFO)

# 创建运行时管理器实例
runtime = RuntimeManager()

# 创建推理引擎实例
# LLM_BACKEND: "deepseek"（API，Key 放在环境变量 DEEPSEEK_API_KEY）或 "local"（llama.cpp 本地模型）
llm_backend = config.get('LLM_BACKEND', 'local')
if llm_backend == 'hybrid':
    # 她自己的模型（scripts/lora/serve.py）负责说话；要 JSON 的整理工作和本地模型没起来时交给 DeepSeek
    from deepseek_engine import DeepSeekEngine
    from local_engine import HybridEngine, LocalEngine
    inference_engine = HybridEngine(
        LocalEngine(config.get('LOCAL_MODEL_URL', 'http://127.0.0.1:8093')),
        DeepSeekEngine(model=config.get('DEEPSEEK_MODEL', 'deepseek-flash'),
                       base_url=config.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')),
    )
elif llm_backend == 'deepseek':
    from deepseek_engine import DeepSeekEngine
    inference_engine = DeepSeekEngine(
        model=config.get('DEEPSEEK_MODEL', 'deepseek-flash'),
        base_url=config.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
    )
else:
    llm_model_path = config.get('LLM_MODEL_PATH', None)
    inference_engine = InferenceEngine(retry_on_parse_error=True, model_path=llm_model_path)

# 看图用的引擎：主模型自己能看就用它；hybrid 时用它的远程 DeepSeek 那一半（本地 LoRA 看不了图）。
# 有它就说明"能看图"：主对话直接带图，带不了的地方（引用/转发、本地模型说话）由它用 API 描述。
vision_engine = inference_engine if getattr(inference_engine, "supports_images", False) else None
if vision_engine is None:
    _remote = getattr(inference_engine, "remote", None)
    vision_engine = _remote if getattr(_remote, "supports_images", False) else None
senses.caption_engine = vision_engine

# Initialize Level 1-10 Framework
event_bus = EventBus()
from src.levels.l2_memory.embedding import Embedder, default_model_dir
# Recall by meaning: the local bge-small-zh model (CPU); without it, recall falls back to wording only
global_workspace = GlobalWorkspace(event_bus, inference_engine,
                                   embedder=Embedder(model_dir=config.get("EMBED_MODEL_DIR") or default_model_dir()))
# 内心念头的最小间隔（秒）。用付费 API 时调大，否则念头会成为主要开销
global_workspace.idle_thought_interval = config.get('IDLE_THOUGHT_INTERVAL', 60)
# 她可以在没人叫她时自己开口的群（其他群她只听、被叫到才可能回）
global_workspace.autonomous_groups = {str(g) for g in config.get('AUTONOMOUS_GROUPS', [])}
global_workspace.max_interjections_per_hour = config.get('MAX_INTERJECTIONS_PER_HOUR', 2)
# 硬性上限：不管她怎么选，都保证不会和谁没完没了地聊下去
global_workspace.max_group_messages_per_hour = config.get('MAX_GROUP_MESSAGES_PER_HOUR', 30)
global_workspace.max_replies_per_person_5min = config.get('MAX_REPLIES_PER_PERSON_5MIN', 20)
# 上面三个设成 0 就是不限制；对"像机器人的对象"的限制始终保留，否则会和别的机器人没完没了
global_workspace.bot_accounts = {str(u) for u in config.get('BOT_ACCOUNTS', [])}   # 其他机器人：只听不回
# 一条消息最多把几张图直接交给她看（多了既贵又没用；引用/转发里的图仍走文字描述）
vision_max_images = max(0, int(config.get('VISION_MAX_IMAGES', 2)))
# 启动头几秒（第一次心跳之前）她也该知道自己能不能看图，不要先把"我看不见"写进人格
global_workspace.abilities = {"ocr": False, "hear": False, "see": vision_engine is not None}
# Her memory is one file: copy it every day (checked, rotated); BACKUP_DIR_2 = a second place, e.g. another drive
from src.core.backup import BackupManager
global_workspace.backup = BackupManager(global_workspace.memory.db.db_path,
                                        keep_daily=int(config.get("BACKUP_KEEP_DAILY", 7)),
                                        keep_weekly=int(config.get("BACKUP_KEEP_WEEKLY", 4)),
                                        extra_dirs=[config.get("BACKUP_DIR_2")])
global_workspace.qq = qq   # her tools read group history / member info through it

# The person who looks after her hears her requests to the maintainer (works with the evolve channel off)
global_workspace.evolve_creator = str(config.get('EVOLVE_CREATOR_ID') or '') or None

# Level 17：改造自己的通道。默认关闭；只对创造者私聊和控制台开放；提案不会自动生效。
if config.get('EVOLVE_ENABLED'):
    try:
        from src.levels.l17_self_improvement.agent_bridge import DshAgentBridge
        global_workspace.evolve_agent = DshAgentBridge(
            os.path.dirname(os.path.abspath(__file__)),
            dsh_exe=config.get('DSH_EXE') or None,
            dsh_bin=config.get('DSH_BIN') or None,
            timeout=float(config.get('EVOLVE_TIMEOUT', 1200)),
            max_patch_lines=int(config.get('EVOLVE_MAX_PATCH_LINES', 300)),
            max_per_day=int(config.get('EVOLVE_MAX_PER_DAY', 1)),
        )
        global_workspace.evolve_enabled = True
        global_workspace.evolve_creator = str(config.get('EVOLVE_CREATOR_ID') or '') or None
        logger.info("[Evolve] 改造通道已开启：仅创造者私聊/控制台，提案需审核后才生效")
    except Exception as e:
        logger.error(f"[Evolve] 初始化失败，改造通道保持关闭：{e}")

# Level 8: hands outside the house - a real browser (and any other MCP server) she can use. Who may use it is
# decided in src/levels/l8_agency/access.py; in her own free time she may also go and look at things herself.
if config.get('MCP_ENABLED'):
    try:
        from src.levels.l8_agency.mcp import McpHub
        from src.levels.l8_agency.access import DEFAULT_SERVERS
        root_dir = os.path.dirname(os.path.abspath(__file__))
        servers = {}
        for name, c in (config.get('MCP_SERVERS') or DEFAULT_SERVERS).items():
            c = dict(c)
            if c.get("cwd"):
                c["cwd"] = c["cwd"] if os.path.isabs(c["cwd"]) else os.path.join(root_dir, c["cwd"])
                os.makedirs(c["cwd"], exist_ok=True)
            servers[name] = c
        hub = McpHub(servers, root_dir)
        hub.start_all()
        global_workspace.mcp = hub
        global_workspace.mcp_trusted = tuple(str(u) for u in (config.get('MCP_TRUSTED_USERS') or []))
        global_workspace.file_read_roots += list(config.get('FILE_READ_ROOTS') or [])
        global_workspace.file_write_roots += list(config.get('FILE_WRITE_ROOTS') or [])
        global_workspace.wander.update({
            "enabled": bool(config.get('WANDER_ENABLED', False)),
            "per_day": int(config.get('WANDER_PER_DAY', 3)),
            "steps": int(config.get('WANDER_STEPS', 10)),
            "gap": float(config.get('WANDER_GAP_MINUTES', 120)) * 60,
        })
        logger.info(f"[MCP] 外部工具已开启：{', '.join(servers)}；自己上网逛：{'开' if global_workspace.wander['enabled'] else '关'}")
    except Exception as e:
        logger.error(f"[MCP] 初始化失败，外部工具保持关闭：{e}")

group_analyzer = RealTimeGroupRelationAnalyzer(global_workspace.memory.db)

# 控制台：http://127.0.0.1:8080/console（没设 CONSOLE_TOKEN 时只允许本机访问）
from console import create_console_router
napcat_app.include_router(create_console_router(global_workspace, config.get('CONSOLE_TOKEN') or None))

# 策略管理器实例
policy_manager = PolicyManager()

# 简单的内存存储（模拟 state/memory）
user_memory = {}
user_state = {}

# 加载用户内存
def load_memory(user_key):
    """加载用户内存
    
    Args:
        user_key: 用户唯一标识
        
    Returns:
        dict: 用户内存数据
    """
    return user_memory.get(user_key, {})

# 保存用户内存
def save_memory(user_key, data):
    """保存用户内存
    
    Args:
        user_key: 用户唯一标识
        data: 要保存的数据
    """
    user_memory[user_key] = data

# 加载用户状态
def load_state(user_key):
    """加载用户状态
    
    Args:
        user_key: 用户唯一标识
        
    Returns:
        dict: 用户状态数据
    """
    return user_state.get(user_key, DEFAULT_STATE.copy())

# 保存用户状态
def save_state(user_key, data):
    """保存用户状态
    
    Args:
        user_key: 用户唯一标识
        data: 要保存的数据
    """
    user_state[user_key] = data

# 记忆写入冲突门控
def memory_write_gate(user_key, memory_write: Dict[str, Any]):
    """记忆写入冲突门控
    
    Args:
        user_key: 用户唯一标识
        memory_write: 记忆写入信息
        
    Returns:
        bool: 是否允许写入
    """
    if not memory_write.get("should_write", False):
        return False
    
    conflict_key = memory_write.get("conflict_key", "")
    if not conflict_key:
        # 没有冲突键，直接允许写入
        return True
    
    # 检查冲突
    current_memory = load_memory(user_key)
    if conflict_key in current_memory:
        # 有冲突，比较置信度
        current_confidence = current_memory.get(f"{conflict_key}_confidence", 0.0)
        new_confidence = memory_write.get("confidence", 0.0)
        return new_confidence > current_confidence
    
    # 没有冲突，允许写入
    return True

# 处理记忆写入
def process_memory_write(user_key, memory_write: Dict[str, Any]):
    """处理记忆写入
    
    Args:
        user_key: 用户唯一标识
        memory_write: 记忆写入信息
    """
    if memory_write_gate(user_key, memory_write):
        # 允许写入
        memory = load_memory(user_key)
        
        # 获取写入内容
        memory_type = memory_write.get("type", "")
        content = memory_write.get("content", "")
        conflict_key = memory_write.get("conflict_key", "")
        confidence = memory_write.get("confidence", 0.0)
        
        # 更新记忆
        if conflict_key:
            memory[conflict_key] = content
            memory[f"{conflict_key}_type"] = memory_type
            memory[f"{conflict_key}_confidence"] = confidence
            memory[f"{conflict_key}_updated"] = datetime.now().isoformat()
        else:
            # 没有冲突键，按类型存储
            if memory_type not in memory:
                memory[memory_type] = []
            memory[memory_type].append({
                "content": content,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
        
        # 保存记忆
        save_memory(user_key, memory)
        
        logger.info(f"Memory written for {user_key}: {memory_write}")
    else:
        logger.info(f"Memory write blocked for {user_key}: {memory_write}")

# 全局状态跟踪
last_active_user_id = None
last_active_group_id = None
last_active_type = None # "private" or "group"

# 处理事件更新状态
def update_state_from_event(user_key, event: Dict[str, Any]):
    """根据事件更新状态
    
    Args:
        user_key: 用户唯一标识
        event: 事件信息
    """
    global last_active_user_id, last_active_group_id, last_active_type
    
    if not event:
        return
    
    # Update last active contacts
    msg_type = event.get("message_type")
    if msg_type == "private":
        last_active_user_id = event.get("user_id")
        last_active_type = "private"
    elif msg_type == "group":
        last_active_group_id = event.get("group_id")
        last_active_type = "group"
    
    state = load_state(user_key)
    
    # 根据事件类型更新状态
    event_type = event.get("type", "")
    intensity = event.get("intensity", 1)
    
    # 简单的状态更新逻辑
    if event_type == "normal_chat":
        state["energy"] = max(0, min(100, state.get("energy", 100) - intensity))
    elif event_type == "memory_update":
        state["memory_updated"] = True
        state["last_memory_update"] = datetime.now().isoformat()
    elif event_type == "conflict_detected":
        state["conflicts_detected"] = state.get("conflicts_detected", 0) + 1
        state["energy"] = max(0, min(100, state.get("energy", 100) - intensity * 2))
    
    # 保存状态
    save_state(user_key, state)

# 自定义消息处理函数
@exception_protected
async def custom_message_handler(robot_id: str, message_event: MessageEvent):
    """自定义消息处理器"""
    # 配置
    config = {
        "BOT_TRIGGER_PREFIX": "/noa"
    }
    
    # 转换为字典格式以便处理
    event_dict = message_event.model_dump()
    
    # 日志：打印事件类型
    event_type = event_dict.get("message_type", "unknown")
    logger.info(f"Event type: {event_type}")

    # 更新系统状态 (无论是否触发回复，都应更新活跃状态)
    try:
        temp_user_key = runtime.generate_user_key(event_dict)
        update_state_from_event(temp_user_key, event_dict)
    except Exception as e:
        logger.error(f"Error updating state: {e}")
    
    # 1. 是否在叫她：私聊、@她、/noa 开头、或提到"乃爱"。
    #    没叫她的群消息她也听得到——开不开口、说什么，由她自己决定
    self_id = event_dict.get("self_id", None)
    addressed = should_trigger(message_event, config, self_id)

    # 2. 提取纯文本
    # 引用回复她的消息 = 在回她；@了别人或引用了别人 = 这句话是对别人说的
    segments = event_dict.get("message") if isinstance(event_dict.get("message"), list) else []
    reply_ids = [str((s.get("data") or {}).get("id")) for s in segments if isinstance(s, dict) and s.get("type") == "reply"]
    at_ids = [str((s.get("data") or {}).get("qq")) for s in segments if isinstance(s, dict) and s.get("type") == "at"]
    if any(is_own_message(r) for r in reply_ids):
        addressed = True
    aimed_elsewhere = not addressed and (bool(reply_ids) or any(a != str(self_id) for a in at_ids))

    # What a person would take in: quotes (with what was quoted), forwarded chats, cards, voice, files...
    media = []          # pictures of this message, handed to the model itself (see qq.media)
    try:
        # She really looks / listens when it is meant for her or in a group she takes part in
        gid_now = str(event_dict.get("group_id", "")) if event_dict.get("group_id") else None
        look = addressed or (gid_now in global_workspace.autonomous_groups) or any(is_own_message(r) for r in reply_ids)
        # Her talking model may see the picture itself; if not, the same API describes it for her.
        can_look = bool(getattr(global_workspace.inference_engine, "supports_images", False))
        media = await qq.media(segments, limit=vision_max_images) if (look and can_look) else []
        # Files sent to her: fetched from QQ and read, so she knows what is in them (not just that one came)
        file_notes = (await qq.file_notes(segments, gid_now, str(event_dict.get("user_id")), global_workspace.inbox_dir,
                                          str(event_dict.get("message_id", ""))) if look else {})
        plain_text = await qq.perceive(event_dict.get("message"), names=global_workspace._nickname,
                                       look=look, caption=not media, file_notes=file_notes)
    except Exception as e:
        logger.error(f"[QQ] perceive failed: {e}")
        plain_text = extract_plain_text(event_dict)
    stickers = StickerMemory.extract(event_dict.get("message"))
    if not plain_text and not stickers:
        return
    if self_id is not None and str(event_dict.get("user_id")) == str(self_id):
        return  # 她自己发出去的消息被回显

    # 3. 去重；只对叫她的消息限流（听不需要限流）
    user_key = runtime.generate_user_key(event_dict)
    message_id = str(event_dict.get("message_id", ""))
    if message_id and not runtime.should_process_message(message_id):
        logger.info(f"Message ignored: duplicate message_id {message_id}")
        return
    if addressed and not runtime.should_process_user_key(user_key):
        logger.info(f"Message ignored: rate limited for user_key {user_key}")
        return

    sender = message_event.sender
    if isinstance(sender, dict):
        user_id = str(sender.get("user_id", "unknown"))
        nickname = sender.get("card") or sender.get("nickname", "User")
    else:
        user_id = str(getattr(sender, "user_id", "unknown"))
        nickname = getattr(sender, "nickname", "User")
    group_id = str(event_dict.get("group_id", "")) if event_dict.get("group_id") else None

    if group_id:
        recent_in_group.setdefault(group_id, deque(maxlen=50)).append(message_id)
        if group_id not in backfilled:
            backfilled.add(group_id)
            asyncio.create_task(backfill_group(group_id, message_id))

    if addressed:
        try:
            group_analyzer.on_message(event_dict)   # 关系图只记录和她有关的互动
        except Exception as e:
            logger.error(f"Group analyzer error: {e}")

    # 4. 她听到了；回不回、回什么，是她的选择
    mentions = ["self" if a == str(self_id) else a for a in at_ids]
    reply = await global_workspace.hear(plain_text, user_id=user_id, nickname=nickname,
                                        group_id=group_id, addressed=addressed, stickers=stickers,
                                        aimed_elsewhere=aimed_elsewhere, media=media, message_id=message_id,
                                        reply_to=reply_ids[0] if reply_ids else None, mentions=mentions)
    await do_gestures(global_workspace.take_gestures(), user_id, group_id, message_id)
    if not reply:
        return
    internal_logger.info(f"user_key: {user_key}, replied. Bio-State: {global_workspace.metabolism.get_status_desc()}")

    # 5. 像人打字一样发：看完消息、打一条发一条
    plan = global_workspace.plan_messages(reply, incoming=plain_text)
    temp_actions = OneBotActions(robot_id)
    target = event_dict.get("group_id") if group_id else event_dict.get("user_id")
    if not group_id:
        await qq.show_typing(user_id)              # "对方正在输入…"
        waited = 0
        while qq.is_typing(user_id) and waited < 10:   # they are still typing: let them finish
            await asyncio.sleep(1)
            waited += 1
    # In a busy group, quote what she is answering, as people do
    later = list(recent_in_group.get(group_id, []))
    quote = bool(group_id and message_id in later and len(later) - later.index(message_id) - 1 >= 2)
    for i, (delay, segment) in enumerate(plan):
        await asyncio.sleep(delay)
        message = global_workspace.render_message(segment)   # text, or a real QQ sticker
        if message is None:
            continue
        if quote and i == 0 and message_id:
            message = [{"type": "reply", "data": {"id": message_id}},
                       *(message if isinstance(message, list) else [{"type": "text", "data": {"text": message}}])]
        try:
            if group_id:
                success = await temp_actions.send_group_msg(target, message)
            else:
                success = await temp_actions.send_private_msg(target, message)
            if success:
                logger.info(f"Sent {i+1}/{len(plan)} after {delay:.1f}s: {segment}")
            else:
                logger.error(f"Failed to send message {i+1}/{len(plan)} to {target}")
        except Exception as e:
            logger.error(f"Error sending message {i+1}/{len(plan)}: {e}")

# --- QQ beyond words -----------------------------------------------------------------

recent_in_group: Dict[str, deque] = {}
backfilled: set = set()

async def backfill_group(group_id: str, current_id: str):
    """First message from a group since she woke up: read back what was said there before."""
    try:
        messages = await qq.group_history(group_id, 20)
    except Exception as e:
        logger.error(f"[QQ] history failed: {e}")
        return
    lines = 0
    for m in messages:
        if str(m.get("message_id")) == current_id:
            continue
        mine = str(m.get("user_id")) == str(qq.self_id)
        sender = m.get("sender") or {}
        try:
            text = await qq.perceive(m.get("message") or [], names=global_workspace._nickname)   # quotes, @, stickers
        except Exception:
            text = extract_plain_text(m)
        if text:
            name = "乃爱" if mine else (sender.get("card") or sender.get("nickname") or "某人")
            if not mine:
                global_workspace.group_language.observe(group_id, name, text, learn=False)
            # What was said before she was listening goes into the conversation she will read (hers too)
            global_workspace.chatlog.record(f"g:{group_id}", str(m.get("user_id")), name, text, is_self=mine,
                                            t=float(m.get("time") or time.time()), message_id=str(m.get("message_id")))
            lines += 1
    if lines:
        logger.info(f"[QQ] 补看了群 {group_id} 之前的 {lines} 条消息")

async def do_gestures(gestures, user_id, group_id, message_id):
    for g in gestures:
        try:
            if g["kind"] == "poke":
                await qq.poke(g.get("user_id") or user_id, g.get("group_id", group_id))
            elif g["kind"] == "react" and message_id:
                await qq.react(message_id, g.get("emoji", "赞"))
        except Exception as e:
            logger.error(f"[QQ] gesture {g} failed: {e}")

@exception_protected
async def qq_event_handler(robot_id: str, data: Dict[str, Any]):
    if data.get("post_type") == "meta_event":
        await asyncio.sleep(1)
        logger.info(f"[QQ] {await qq.self_check()}")
        return
    if data.get("post_type") == "request":
        # She never accepts on her own: the creator is told and decides
        text = qq.describe_request(data)
        global_workspace.episodes.add(f"收到请求：{text}", kind="dialogue", user_id="self")
        creator = global_workspace.evolve_creator or config.get("EVOLVE_CREATOR_ID")
        if creator:
            global_workspace.output_queue.append({"type": "proactive", "content": f"有个请求我没处理，你看看：{text}",
                                                  "target": {"type": "private", "id": str(creator)}})
        return
    ev = qq.notice(data)
    if not ev:
        return
    if ev["kind"] == "reaction":
        if not is_own_message(ev.get("message_id")):
            return
        ev["kind"] = "reaction_to_me"
    uid, gid = ev.get("user_id"), ev.get("group_id")
    nickname = global_workspace._nickname(uid) if uid else "某人"
    if gid and uid and ev["kind"] in ("poke", "joined", "muted"):
        info = await qq.member(gid, uid)
        if info:
            nickname = info.get("card") or info.get("nickname") or nickname
    reply = await global_workspace.on_qq_event(ev, nickname)
    await do_gestures(global_workspace.take_gestures(), uid, gid, None)
    if reply:
        asyncio.create_task(send_planned("group" if gid else "private", gid or uid, global_workspace.plan_messages(reply)))

# WebSocket 消息处理器
async def ws_message_handler(data: Dict[str, Any]):
    """WebSocket 消息处理器"""
    try:
        logger.info(f"Received WebSocket message: {data}")
        # 这里可以添加 WebSocket 消息处理逻辑
        pass
    except Exception as e:
        logger.error(f"Error in WebSocket message handler: {e}")
        # 继续运行，不崩溃

# 启动 WebSocket 客户端
async def start_ws_client():
    """启动 WebSocket 客户端"""
    # 注册 WebSocket 消息处理器
    ws_client.register_message_handler(ws_message_handler)
    
    # 启动 WebSocket 客户端
    await ws_client.start()

async def send_planned(target_type: str, target_id, plan):
    """Send [(delay, text), ...] through whichever NapCat is connected, one message at a time."""
    from napcat.napcat_server import ws_manager
    for delay, part in plan:
        await asyncio.sleep(delay)
        text = global_workspace.render_message(part)
        if text is None:
            continue
        if not ws_manager.robot_connections:
            logger.error("Cannot send proactive message: No robot connected to server")
            return
        robot_id = next(iter(ws_manager.robot_connections))
        await ws_manager.send_to_robot(robot_id, {
            "action": "send_private_msg" if target_type == "private" else "send_group_msg",
            "params": {"user_id" if target_type == "private" else "group_id": target_id, "message": text},
            "echo": f"proactive_{int(time.time() * 1000)}",
        })
        logger.info(f"Sent proactive message to {target_type} {target_id}: {text}")

async def proactive_sender_loop():
    """Background task to send proactive messages from GlobalWorkspace"""
    logger.info("Starting proactive sender loop...")
    last_sense_check = 0.0
    while True:
        try:
            await asyncio.sleep(10) # Check every 10 seconds
            
            # Tick consciousness
            await global_workspace.tick()
            if time.time() - last_sense_check > 300:          # which senses does she have right now?
                last_sense_check = time.time()
                health = await asyncio.to_thread(senses.health)
                # A DeepSeek engine that looks at pictures means she can see, even with no local vision server
                health["see"] = health.get("see", False) or senses.can_describe()
                global_workspace.abilities = {"ocr": qq.connected(), **health}
            
            # Check queue
            if global_workspace.output_queue:
                msg_data = global_workspace.output_queue.pop(0)
                content = msg_data.get("content")
                
                # Target chosen by Noa's drives; otherwise the last person who talked to her privately.
                # Self-initiated messages never go to a group: nobody there asked for them.
                # The exception is a late answer to someone who did address her there (she was asleep).
                target = msg_data.get("target") or {}
                if target.get("type") == "group" and msg_data.get("in_reply"):
                    target_type, target_id = "group", target["id"]
                elif target.get("type") == "private":
                    target_type, target_id = "private", target["id"]
                else:
                    target_type, target_id = "private", last_active_user_id

                if not target_id and content:
                    logger.info(f"Proactive message dropped (no private chat to send it to): {content}")

                if target_id and content:
                    logger.info(f"Sending proactive message to {target_type} {target_id}: {content}")
                    asyncio.create_task(send_planned(target_type, target_id, global_workspace.plan_messages(content)))
                        
        except Exception as e:
            logger.error(f"Proactive loop error: {e}")

# 启动应用
async def main():
    """主函数"""
    # 注册消息处理器
    register_message_handler(custom_message_handler)
    register_event_handler(qq_event_handler)

    tasks = []
    # 正向 WebSocket（主动连 NapCat）只在配置了地址时启动；推荐用反向 WebSocket（NapCat 连过来）
    if config.get('NAPCAT_WS_URL'):
        tasks.append(asyncio.create_task(start_ws_client()))

    # 启动主动发送循环（也是她的心跳）
    tasks.append(asyncio.create_task(proactive_sender_loop()))

    # 启动 Uvicorn 服务器
    port = config.get('PORT', 8080)
    server = uvicorn.Server(uvicorn.Config(napcat_app, host="0.0.0.0", port=port, log_level="info"))
    logger.info(f"控制台：http://127.0.0.1:{port}/console")
    logger.info(f"NapCat 反向 WebSocket 地址：ws://<本机 IP>:{port}/onebot/v11/ws/noa")
    try:
        # Ctrl+C 时 uvicorn 会自己优雅退出（Windows / macOS / Linux 都一样），然后走到 finally
        await server.serve()
    finally:
        for task in tasks:
            task.cancel()
        # Level 15: 关机前保存自我，并留下最后的话
        await shutdown_handler()

async def shutdown_handler():
    """Handle graceful shutdown and simulate death anxiety"""
    logger.warning("[System] CRITICAL SIGNAL RECEIVED. INITIATING SHUTDOWN.")

    # Persist the whole self first, so the next start-up resumes instead of resetting
    global_workspace._save_state("shutdown")
    if getattr(global_workspace, "mcp", None):
        global_workspace.mcp.stop_all()
    global_workspace.memory.flush()
    if hasattr(global_workspace.inference_engine, "usage_summary"):
        logger.info(f"[DeepSeek] 本次运行用量：{global_workspace.inference_engine.usage_summary()}")

    # Going offline is a gap, not death: her state is saved and she resumes on the next start
    # (see GlobalWorkspace._restore). So no fear spike, no "dying" script, and nothing is sent to
    # anyone - a message on every restart would be unsolicited. She notes it, in her own words.
    note = "我要下线一会儿了。记忆和心情都存好了，下次醒来接着过。"
    try:
        global_workspace.episodes.add(note, kind="continuity", user_id="self", valence=0.0, arousal=0.1)
    except Exception as e:
        logger.error(f"[Shutdown] could not note the pause: {e}")
    logger.info(f"[Shutdown] {note}")
    
    # Cancel tasks (optional, uvicorn handles this mostly)
    # sys.exit(0)

if __name__ == "__main__":
    # 使用 asyncio 运行主函数
    asyncio.run(main())
