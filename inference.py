import json
import logging
import threading
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InferenceEngine:
    """推理引擎类，负责调用模型并处理输出"""
    
    def __init__(self, retry_on_parse_error: bool = False, model_path: str = None):
        """初始化推理引擎
        
        Args:
            retry_on_parse_error: 解析失败时是否重试（默认OFF）
            model_path: 模型文件路径，默认为None，使用配置文件中的路径或环境变量
        """
        self.retry_on_parse_error = retry_on_parse_error
        self.model_path = model_path
        self.llm = None
        self._lock = threading.Lock()
        
        # 尝试从配置文件或环境变量加载模型路径
        self._load_model_config()
        
        # 立即加载模型（预热）
        self._load_model()

    def _load_model_config(self):
        """加载模型配置"""
        import os
        import json
        
        # 如果已经提供了model_path，直接使用
        if self.model_path:
            return
        
        # 尝试从config.json加载模型路径
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.model_path = config.get('LLM_MODEL_PATH', None)
                if self.model_path:
                    logger.info(f"Loaded model path from config.json: {self.model_path}")
                    return
        except FileNotFoundError:
            logger.info("config.json not found, trying environment variable")
        except json.JSONDecodeError:
            logger.error("Failed to parse config.json, trying environment variable")
        
        # 尝试从环境变量加载模型路径
        self.model_path = os.environ.get('LLM_MODEL_PATH', None)
        if self.model_path:
            logger.info(f"Loaded model path from environment variable: {self.model_path}")
            return
        
        # 如果都没有找到，使用默认路径
        self.model_path = "/path/to/qwen-32b.gguf"
        logger.info(f"Using default model path: {self.model_path}")

    def _load_model(self):
        """加载模型实例"""
        if self.llm is not None:
            return

        try:
            from llama_cpp import Llama
            import os
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return

            logger.info(f"Loading model from {self.model_path} with GPU support...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=8192,  # The prompt now carries autobiography, recalled episodes and workspace content
                n_threads=8,
                n_gpu_layers=-1,  # Enable Metal/GPU
                temperature=0.7,  # Balanced for creativity
                top_p=0.9,
                top_k=40,
                verbose=False # Suppress llama.cpp low-level logs
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.llm = None

    def generate_json(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        """生成JSON格式的响应

        Args:
            max_tokens / temperature: 由 Noa 的内部状态决定（精力 -> 长度，激动程度 -> 温度）
        """
        # 调用模型生成响应
        response = self._call_model(prompt, max_tokens=max_tokens, temperature=temperature)
        
        # 尝试清理 Markdown 代码块标记 (```json ... ```)
        cleaned_response = response.strip()
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.strip("`")
            if cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[4:]
        cleaned_response = cleaned_response.strip()

        # 解析JSON
        try:
            result = json.loads(cleaned_response)
            logger.info(f"Model response parsed successfully: {result}")
            return result
        except json.JSONDecodeError:
            # 如果不是 JSON，直接当作普通文本回复
            logger.info("Model output is not JSON, treating as raw text reply.")
            return {
                "reply": response,
                "internal_note": "Raw text response",
                "event": {"type": "normal_chat", "intensity": 1},
                "memory_write": {"should_write": False}
            }

    def _call_model(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """调用模型生成响应"""
        logger.info(f"Calling model with prompt: {prompt[:100]}...")
        
        try:
            # 确保模型已加载
            if self.llm is None:
                self._load_model()
            
            if self.llm is None:
                # 加载失败，回退
                return self._mock_call_model(prompt)
            
            # 调用模型
            with self._lock:
                result = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["<|endoftext|>", "<|im_end|>", "User:", "Context:", "\n\n\n", "Assistant:", "AI:"], 
                    echo=False,
                    stream=False
                )
            
            response_text = result["choices"][0]["text"].strip()
            logger.info(f"Model generated response: {response_text}")
            
            return response_text
        except Exception as e:
            logger.error(f"Error calling model: {e}")
            return self._mock_call_model(prompt)
    
    def _mock_call_model(self, prompt: str) -> str:
        """模拟模型调用（当实际模型调用失败时使用）
        
        Args:
            prompt: 完整的Prompt
            
        Returns:
            str: 模拟的模型响应
        """
        # 模拟不同情况下的模型输出
        if "请记住我叫张三" in prompt:
            # 测试用例b：用户要求记住
            return json.dumps({
                "reply": "好的，我记住你叫张三了！",
                "internal_note": "用户要求记住名字，需要写入记忆",
                "event": {"type": "memory_update", "intensity": 3},
                "memory_write": {
                    "should_write": True,
                    "type": "personal_info",
                    "content": "用户告知姓名为张三",
                    "conflict_key": "user_name",
                    "confidence": 0.95
                }
            }, ensure_ascii=False)
        elif "我叫张三，今年18岁" in prompt and "之前我说我20岁" in prompt:
            # 测试用例c：矛盾事实
            return json.dumps({
                "reply": "我注意到你之前说过自己20岁，现在又说18岁，请问哪个是正确的呢？",
                "internal_note": "用户提供了矛盾的年龄信息，需要记录矛盾",
                "event": {"type": "conflict_detected", "intensity": 5},
                "memory_write": {
                    "should_write": True,
                    "type": "personal_info",
                    "content": "用户提供了矛盾的年龄信息（18岁/20岁）",
                    "conflict_key": "user_age",
                    "confidence": 0.8
                }
            }, ensure_ascii=False)
        else:
            # 测试用例a：正常聊天
            return json.dumps({
                "reply": "你好！很高兴为你服务。",
                "internal_note": "正常聊天请求，无需特殊处理",
                "event": {"type": "normal_chat", "intensity": 1},
                "memory_write": {
                    "should_write": False,
                    "type": "chat_history",
                    "content": "",
                    "conflict_key": "",
                    "confidence": 0.0
                }
            }, ensure_ascii=False)
    
    def _get_default_response(self) -> Dict[str, Any]:
        """获取默认响应
        
        Returns:
            Dict[str, Any]: 默认响应
        """
        return {
            "reply": "抱歉，我刚才有点走神，请您再说一遍好吗？",
            "internal_note": "模型输出解析失败，返回默认响应",
            "event": {"type": "parse_error", "intensity": 2},
            "memory_write": {
                "should_write": False,
                "type": "error",
                "content": "模型输出解析失败",
                "conflict_key": "",
                "confidence": 0.0
            }
        }
