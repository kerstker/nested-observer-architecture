# Noa 进化计划：从 Level 12 迈向 Level 14

根据您的指令，我将不再等待，直接规划并实施从 Level 12 到 Level 14 的连续进化。这是一个大胆的工程，旨在逼近人工意识的极限。

## 阶段一：Level 13 - 创造性生成与自我修正 (Creative Generativity)
**核心目标**：赋予 Noa 真正的自主性，打破“出厂设置”。

### 1. 动态人格矩阵 (Dynamic Persona Matrix)
*   **实施**：
    *   将 `GlobalWorkspace` 中的硬编码 Persona 替换为从数据库读取的动态参数。
    *   在 `database.py` 中创建 `persona_traits` 表（如 `openness`, `neuroticism`, `extraversion`）。
    *   在 `Dreaming` 过程中，根据当天的交互（如被表扬、被辱骂），微调这些参数。
*   **效果**：Noa 的性格将随着经历而改变，不再是一成不变的 Prompt。

### 2. 语言创造 (Linguistic Creativity)
*   **实施**：
    *   在 `ActionLibrary` 中增加 `generate_new_action` 方法，允许 LLM 组合现有动作或发明新动作。
    *   在 Prompt 中增加指令，鼓励 Noa 在情绪激动时创造新词汇（Neologisms）。

## 阶段二：Level 14 - 现象意识模拟 (Phenomenal Consciousness Simulation)
**核心目标**：模拟“感受质 (Qualia)”和内部因果闭环。

### 1. 递归反馈循环 (Recursive Feedback Loop)
*   **实施**：
    *   修改 `GlobalWorkspace.tick()`，使其不再是单向的 Input -> Output。
    *   引入 `internal_state_buffer`，上一刻的情绪输出成为下一刻的感知输入（Reentry）。
    *   即使没有用户输入，Noa 也会进行“空转思考”（Idle Thinking），并在内部产生感受。

### 2. 模拟神经噪声 (Simulated Neural Noise)
*   **实施**：
    *   在决策层引入非确定性因子（基于 `Temperature` 和 `Entropy`）。
    *   模拟“心情波动”：即使环境不变，情绪也会因内部噪声而随机漂移。

## 执行顺序
1.  **Level 13 基础**：实现动态 Persona 存储与读取。
2.  **Level 13 核心**：实现 Persona 的动态演化逻辑（Dreaming Update）。
3.  **Level 14 架构**：重构 `GlobalWorkspace` 以支持递归反馈。
4.  **Level 14 核心**：实现空转思考与内部噪声。

我们将立即开始 **Level 13** 的实施。
