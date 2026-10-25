# Level 11: Meta-Cognition & Agency Upgrade Plan

This plan aims to bridge the "Philosophical Gap" by implementing Active Agency (Will), Inner Monologue (Thought), and Dreaming (Subconscious Processing).

## 1. Active Agency (The "Will" to Act)
**Goal**: Transform Noa from a passive responder to an active agent who initiates contact based on internal desires (Loneliness, Boredom).

### Implementation:
*   **GlobalWorkspace (`global_workspace.py`)**:
    *   Add `self.output_queue` to store proactive messages.
    *   Modify `tick()` loop:
        *   Monitor `Social Need` and `Energy`.
        *   If `Social Need > 0.7` AND `Energy > 30` AND `Time since last chat > 1 hour`:
            *   Trigger `_generate_proactive_message()`.
            *   Push result to `output_queue`.
*   **Application Layer (`app.py`)**:
    *   Implement a background `async` task `proactive_sender_loop()`.
    *   Poll `global_workspace.output_queue` every 10 seconds.
    *   If a message exists, send it to the user/group via `OneBotActions`.

## 2. Inner Monologue (The "Cogito")
**Goal**: Simulate the gap between thought and speech. Noa will "think" before she speaks, allowing for deception, planning, or emotional regulation.

### Implementation:
*   **Prompt Engineering (`global_workspace.py`)**:
    *   Update System Prompt to enforce a `Thought` / `Response` format.
    *   Example Output:
        ```
        Thought: The user seems sad. My energy is low, but I should prioritize comforting them.
        Response: *gently pats head* It's okay, I'm here for you...
        ```
*   **Response Processing**:
    *   Parse the LLM output to separate `Thought` from `Response`.
    *   Log `Thought` to internal logs (Consciousness Stream).
    *   Send only `Response` to the user.

## 3. Dreaming System (The "Subconscious")
**Goal**: Process daily experiences into long-term wisdom during sleep cycles.

### Implementation:
*   **Sleep Logic (`metabolism.py` & `global_workspace.py`)**:
    *   When `Bio-State` switches to `Sleeping`:
        *   Trigger `_dream_process()`.
        *   Retrieve today's `interactions`.
        *   Ask LLM to summarize key events and update `User Profile` or `Relationships`.
        *   Clear/Archive short-term memory to prepare for the next "day".

## Execution Order
1.  **Refactor Prompt** for Inner Monologue (Immediate intelligence boost).
2.  **Implement Agency** logic in `tick()` and `app.py` (Behavioral change).
3.  **Implement Dreaming** logic (Long-term retention).
