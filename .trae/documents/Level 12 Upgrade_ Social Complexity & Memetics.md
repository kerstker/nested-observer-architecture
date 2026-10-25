# Level 12 深度实施计划：真实的社会表达与性能极致

用户的核心需求是：不仅要有后台的“社会系统”，更要让 Noa **真的能够表达出来**。这意味着她需要主动八卦、评论群友关系、使用流行梗，并对不同声誉的人表现出明显的态度差异。

## 1. 核心表达能力 (Expressive Agency)

### 1.1 主动八卦与社交评论 (Social Commentary)
*   **目标**：Noa 不仅知道 A 和 B 关系好，还要能**主动提起**。
*   **表达方式**：
    *   “哎呀，最近看你和 @XXX 聊得很火热嘛~”
    *   “哼，@XXX 总是欺负人，不想理他了。”
*   **实现**：
    *   在 `tick()` 的主动行为逻辑中，增加 `SocialGossip` 策略。
    *   从 `Social Graph` 中检索高权重的边，生成评论。

### 1.2 模因（梗）的活学活用 (Meme Usage)
*   **目标**：自然地在对话中使用群友常说的梗，而不是生硬的复读。
*   **表达方式**：
    *   用户：“我裂开了” -> Noa：“裂开了+1 喵~”
*   **实现**：
    *   在 `Prompt` 中动态注入 `[Current Group Memes]`。
    *   指令 LLM 尝试使用这些梗。

### 1.3 态度鲜明的声誉反应 (Reputation-based Attitude)
*   **目标**：对“好人”甜美，对“坏人”毒舌。
*   **表达方式**：
    *   高信誉：秒回，语气软萌，多用表情包。
    *   低信誉（如经常调戏）：敷衍，毒舌，甚至拒绝服务。
*   **实现**：
    *   在 Prompt 中注入 `[User Reputation]`。
    *   根据声誉调整 System Prompt 中的 `Tone` 指令。

## 2. 系统实现 (System Implementation)

### 2.1 数据库与社会图谱
*   **升级**：`relationships` 表支持 `user_id_a` <-> `user_id_b`。
*   **逻辑**：在 `GlobalWorkspace` 中增加 `_observe_social_interaction`，分析群聊中谁 At 了谁，更新关系强度。

### 2.2 性能优化 (Performance)
*   **异步写入**：所有 DB 写操作放入 `WriteQueue`，由独立线程处理。
*   **Prompt 缓存**：优化 Meme 和 Social Graph 的注入方式，尽量保持前缀静态。

## 执行顺序
1.  **数据库升级**：支持任意用户关系。
2.  **表达逻辑植入**：修改 Prompt 和 `GlobalWorkspace`，注入社会上下文和模因。
3.  **主动八卦机制**：在 `tick()` 中加入社交评论触发器。
4.  **性能重构**：实现 DB 写操作异步化。
