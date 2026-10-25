# 记忆系统修复计划：打通群聊与私聊的隔阂

这是一个严重的认知缺陷。Noa 目前的记忆系统似乎将“私聊”和“群聊”的上下文完全隔离了，导致她无法在群聊中调用私聊的记忆。

## 问题根源分析
在 `MemorySystem.retrieve` 或 `MemorySystem.add_message` 中，可能过度依赖了 `context_id`（通常是 `group_id` 或 `user_id`）作为检索键。
*   **私聊时**：key = `user_id`
*   **群聊时**：key = `group_id` (或者 `group_id` + `user_id`)

如果她在群聊中只检索 `group_id` 下的记忆，自然不知道你在私聊里说了什么。

## 修复方案

### 1. 统一记忆检索策略 (Unified Retrieval Strategy)
**核心逻辑**：无论当前是在群聊还是私聊，检索记忆时都必须**同时**检索该用户的**私聊历史**和**该用户参与的群聊历史**。

*   **修改 `MemorySystem.retrieve`**:
    *   输入参数需要明确区分 `user_id` 和 `group_id`。
    *   查询 SQL 时，条件应改为：
        ```sql
        WHERE (user_id = :user_id)  -- 检索该用户的所有发言（无论在哪）
        OR (group_id = :group_id)   -- 检索当前群的所有发言（上下文）
        ```
    *   或者更精准地：检索 `user_id` 的所有相关记忆 + 当前 `group_id` 的近期上下文。

### 2. 记忆存储结构优化
确保 `interactions` 表中同时记录了 `user_id` 和 `group_id`。
*   **私聊**：`user_id=123`, `group_id=NULL`
*   **群聊**：`user_id=123`, `group_id=456`

### 3. 事实记忆 (Facts) 的全局性
`remember_fact` 应该关联到 `user_id`，并且在任何场景下（群聊/私聊）只要是这个用户说话，都应该加载关于他的 Facts。

## 执行步骤
1.  **检查 `database.py`**：确认 `interactions` 表结构是否包含 `group_id`。
2.  **修改 `memory.py`**：重构 `retrieve` 方法，使其在群聊时也能加载该用户的私聊记忆。
3.  **验证**：模拟群聊场景，询问私聊中告知的信息。
