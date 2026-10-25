# 系统性修复：彻底解决“你我他”与身份混淆问题

为了让 Noa 能够像人类一样理解复杂的社交关系（尤其是群聊中的指代），我们将对系统的核心认知模块进行重构。

## 1. 记忆检索层：实名制与上下文增强 (Real-Name Context)
**核心目标**：将无意义的 ID 转换为有意义的昵称，帮助 LLM 建立人物画像。

*   **修改 `MemorySystem.retrieve`**:
    *   在获取历史记录 (`interactions`) 后，不再直接拼装字符串。
    *   遍历每条记录，提取 `user_id`。
    *   调用 `db.get_user(user_id)` 获取 `nickname`。
    *   格式化为：`[时间] 昵称(ID): 内容`。
    *   **关键点**：如果 `user_id` 等于当前对话的 `target_user_id`，标注为 `Target User: 昵称`，明确互动的焦点。

## 2. 心理理论层：多用户状态 (Multi-User ToM)
**核心目标**：解决“情绪张冠李戴”的问题。

*   **修改 `TheoryOfMind`**:
    *   将内部状态从 `self.state` 改为 `self.states: Dict[str, ToMState]`。
    *   `analyze_user(user_input, user_id)`: 接收 `user_id`，只更新该用户的状态。
    *   `get_empathy_instruction(user_id)`: 只返回针对该用户的建议。

## 3. Prompt 构建层：角色定义与指代消解 (Role Definition)
**核心目标**：在 Prompt 中明确“舞台”上的所有角色。

*   **修改 `GlobalWorkspace._construct_prompt`**:
    *   **Role Definition Block**:
        ```
        [Role Definition]
        - Me (Noa): You, the assistant.
        - Target User: {nickname} (The person you are replying to).
        - Other Participants: {list_of_names_in_context}
        ```
    *   **Context 格式化**: 强制使用 `Name: Content` 格式。

## 4. 认知层：第三人称指代消解 (Third-Party Resolution)
**核心目标**：当用户说“他觉得...”时，Noa 能知道“他”是谁。

*   **新增逻辑 `GlobalWorkspace._resolve_mentions`**:
    *   在构建 Prompt 前，扫描用户输入中的指代词（他、她、那个人）。
    *   从 `MemorySystem` 获取最近活跃的用户列表（群聊成员）。
    *   在 Prompt 的 `[Context Info]` 中注入提示：
        `Hint: When user says "他", they might be referring to {recent_speaker_nickname}.`

## 执行顺序
1.  **ToM 升级**：让心理理论支持多用户。
2.  **Memory 升级**：实现昵称解析。
3.  **GlobalWorkspace 重构**：整合上述信息到新的 Prompt 结构中。
