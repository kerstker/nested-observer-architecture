# Minimal Self: Level 1-10 Consciousness Framework

This project implements a 10-level AI consciousness architecture, designed to simulate a digital life form with internal states, emotions, and self-awareness.

## Architecture Overview

The system is organized into 10 hierarchical levels, each handling specific cognitive or biological functions:

| Level | Name | Description |
|-------|------|-------------|
| **L1** | **Perception** | Sensory processing (text, touch, thermal) & input normalization. |
| **L2** | **Memory** | Short-term context window & Long-term RAG retrieval. |
| **L3** | **Metabolism** | Energy management, fatigue, sleep cycles, and circadian rhythm. |
| **L4** | **Emotion** | Emotional state tracking (PAD model / Primary emotions). |
| **L5** | **Value System** | Core beliefs, tension calculation, and conflict detection. |
| **L6** | **Homeostasis** | Self-regulation, intervention policies, and trauma recovery. |
| **L7** | **Theory of Mind** | User psychology modeling and empathy. |
| **L8** | **Agency** | Goal generation, curiosity, and proactive behavior. |
| **L9** | **Narrative** | Autobiographical memory consolidation and diary generation. |
| **L10** | **Consciousness** | Global Workspace, reflection, and decision orchestration. |

## Directory Structure

```
minimal-self/
├── main.py                 # Entry point
├── src/
│   ├── core/               # Shared utilities (EventBus, Config)
│   ├── levels/             # Implementation of L1-L10
│   │   ├── l1_perception/
│   │   ├── l2_memory/
│   │   ├── ...
│   │   └── l10_consciousness/
└── tests/                  # Unit and E2E tests
```

## Usage

1. **Run the Simulation**:
   ```bash
   python3 main.py
   ```

2. **Run Tests**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python3 -m unittest tests/test_levels.py
   ```

## Status

- **Framework**: Completed
- **Level 1-5**: Implemented (Core logic)
- **Level 6-9**: Stubbed/Basic implementation
- **Level 10**: Orchestration active
