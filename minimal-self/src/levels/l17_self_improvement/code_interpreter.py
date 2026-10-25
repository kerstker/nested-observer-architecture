from typing import Dict, Any

class CodeInterpreter:
    """
    Level 17: Code Interpreter (DISABLED).

    The previous in-process exec() with an emptied __builtins__ was not a sandbox:
    `().__class__.__bases__[0].__subclasses__()` escapes it, and an infinite loop
    froze the whole bot. Since any QQ group member can steer the LLM into emitting
    `Action: EXECUTE_CODE`, execution is refused until a real isolated sandbox
    (separate container with CPU/memory/network limits) exists.
    """
    enabled = False

    def execute(self, code: str) -> Dict[str, Any]:
        return {
            "success": False,
            "output": "",
            "error": "Code execution is disabled."
        }
