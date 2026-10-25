from typing import Dict, Any, List


class PolicyManager:
    """策略管理器，负责生成和管理策略"""
    
    def __init__(self):
        """初始化策略管理器"""
        # 默认策略规则
        self.default_rules = [
            "保持回复友好、礼貌",
            "保护用户隐私，不泄露敏感信息",
            "遵循事实，不编造信息",
            "对不确定的问题，坦诚告知",
            "保持回复简洁明了，避免冗长",
            "内部笔记（internal_note）不参与用户可见输出"
        ]
    
    def generate_policy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """生成策略
        
        Args:
            context: 上下文信息
            
        Returns:
            Dict[str, Any]: 生成的策略
        """
        # 基于规则生成策略，不消耗额外推理
        policy = {
            "rules": self.default_rules.copy(),
            "context_specific": self._get_context_specific_rules(context)
        }
        
        return policy
    
    def _get_context_specific_rules(self, context: Dict[str, Any]) -> List[str]:
        """获取上下文特定规则
        
        Args:
            context: 上下文信息
            
        Returns:
            List[str]: 上下文特定规则
        """
        specific_rules = []
        
        # 基于消息类型添加特定规则
        message_type = context.get("message_type", "")
        if message_type == "private":
            specific_rules.append("私聊模式下，保持更个性化的回复")
        elif message_type == "group":
            specific_rules.append("群聊模式下，保持公开场合的得体回复")
        
        # 基于用户历史添加特定规则
        user_id = context.get("user_id", "")
        if user_id:
            specific_rules.append(f"针对用户 {user_id}，保持回复的一致性")
        
        return specific_rules
    
    def check_policy_compliance(self, content: str, policy: Dict[str, Any]) -> bool:
        """检查内容是否符合策略
        
        Args:
            content: 要检查的内容
            policy: 策略
            
        Returns:
            bool: 是否符合策略
        """
        # 这里可以添加策略合规检查逻辑
        # 目前简单返回True
        return True
    
    def get_policy_for_logging(self, policy: Dict[str, Any]) -> str:
        """获取用于日志记录的策略表示
        
        Args:
            policy: 策略
            
        Returns:
            str: 用于日志记录的策略字符串
        """
        rules_str = "\n".join(f"- {rule}" for rule in policy.get("rules", []))
        specific_str = "\n".join(f"- {rule}" for rule in policy.get("context_specific", []))
        
        return f"""
通用规则：
{rules_str}

上下文特定规则：
{specific_str}
        """
