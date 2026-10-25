# The First Life 全面优化计划 (修订版)

本计划旨在通过重构核心通信机制、消除代码冗余、引入工程化最佳实践，将系统整体效率提升 30% 以上，并重点解决**日志混乱**和**调试信息泄露**的问题。

## Milestone 1: 核心架构与代码重构 (High Priority)

### 1.1 异步事件总线 (EventBus Refactoring)

* **目标**: 解耦事件发布与处理，消除主线程阻塞。

* **行动**:

  * 重写 `src/core/event_bus.py`，基于 `asyncio` 实现异步 `publish` 和 `subscribe`。

  * 确保所有 Level 模块（感知、记忆、意识）的 `tick` 和回调函数兼容异步模式。

### 1.2 日志治理与系统降噪 (Log Governance) **\[新增]**

* **目标**: 彻底清除底层库（llama.cpp, httpx）的干扰日志，提供干净的控制台输出。

* **行动**:

  * 配置全局 `logging.dictConfig`，将第三方库日志级别设为 `WARNING` 或 `ERROR`。

  * 在 `InferenceEngine` 初始化时设置 `verbose=False` 以静默 `llama.cpp` 的底层输出。

  * 全局扫描并移除所有 `print()` 调试语句，统一替换为结构化 `logger` 调用。

### 1.3 统一入口与代码清理 (Code Cleanup)

* **目标**: 消除多入口冗余，降低维护成本。

* **行动**:

  * 废弃并归档 `minimal-self-noa/app.py` 和 `minimal-self/main.py`。

  * 重构根目录 `app.py`，将 WebSocket 客户端逻辑拆分为独立模块 `src/infrastructure/napcat_client.py`。

## Milestone 2: 性能瓶颈优化 (Performance)

### 2.1 异步推理与缓存

* **目标**: 减少重复计算和 I/O 等待。

* **行动**:

  * **LLM 异步化**: 封装 `InferenceEngine` 调用，确保所有模型推理在独立线程池中执行。

  * **缓存层**: 为 `InferenceEngine` (Embedding/生成) 和 `WebSearchTool` 引入 `LRU Cache`，减少昂贵调用。

  * **数据库优化**: 显式开启 SQLite 的 WAL (Write-Ahead Logging) 模式。

## Milestone 3: 工程效率与验证 (Engineering)

### 3.1 自动化测试体系

* **目标**: 建立回归测试防线。

* **行动**:

  * 引入 `pytest` 和 `pytest-asyncio`。

  * 编写针对 `EventBus` 和 `GlobalWorkspace` 的核心单元测试。

### 3.2 基准测试 (Benchmark)

* **目标**: 量化优化效果。

* **行动**:

  * 编写 `scripts/benchmark_core.py`，对比优化前后的吞吐量（TPS）。

  * 验证优化后系统能否达到 **30% 的效率提升**。

## 执行策略

1. **立即执行**: 优先完成 **日志治理** 和 **EventBus 重构**，解决用户最关心的“调试信息”问题和性能痛点。
2. **并行推进**: 在重构的同时补充测试用例。
3. **最终验证**: 完成基准测试，生成优化报告。

