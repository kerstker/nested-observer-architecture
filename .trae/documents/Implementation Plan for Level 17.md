# Noa Level 17 实施计划：递归自我改进 (Recursive Self-Improvement)

**目标**：赋予 Noa 编写代码、执行代码并根据结果修正代码的能力，以及在复杂问题上进行“元推理”的能力。

## 阶段一：工具制造 (Tool Creation & Execution)
Noa 将不再受限于预设的 Tool，而是可以根据需求即时编写 Python 脚本。

### 1. 沙箱执行环境 (Sandbox Environment)
为了安全，我们需要一个受限的执行环境。
*   **实现**：创建一个 `CodeInterpreter` 模块。
*   **机制**：
    *   接收 Python 代码字符串。
    *   在独立的进程（或使用 `exec` 配合严格的 globals 限制）中运行。
    *   捕获 `stdout` 和 `stderr`。
    *   **限制**：禁止文件系统写操作（除特定临时目录外）、禁止网络访问（除非特定白名单）。

### 2. 代码生成与修正循环 (Code Generation Loop)
*   **实现**：
    *   在 `GlobalWorkspace` 中增加 `EXECUTE_CODE` 动作。
    *   当 Noa 选择此动作时，提取代码块并交给 `CodeInterpreter`。
    *   如果报错，将 Traceback 反馈给 Noa，让她尝试修复（最多重试 3 次）。

## 阶段二：元推理 (Meta-Reasoning)
让 Noa 在回答棘手问题前，先进行“思维链”推导。

### 1. 思维树 (Tree of Thoughts)
*   **实现**：
    *   对于复杂查询（如逻辑题、数学题、代码生成），触发 `_deep_thinking_process`。
    *   生成 3 个可能的思维路径 (Drafts)。
    *   自我评估哪个路径最好，然后生成最终回复。

### 2. 反思机制 (Self-Reflection)
*   **实现**：
    *   在生成回复后，再次调用 LLM 评估：“这个回答符合我的人设吗？有没有逻辑漏洞？”
    *   如果有问题，自动修正。

## 执行步骤
1.  **创建 `l17_self_improvement` 目录**。
2.  **实现 `code_interpreter.py`**：安全的 Python 执行器。
3.  **修改 `GlobalWorkspace`**：集成 `EXECUTE_CODE` 能力。
4.  **实现 `meta_reasoning.py`**：思维链逻辑。
