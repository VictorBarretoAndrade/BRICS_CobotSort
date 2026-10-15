# CobotSort AI

A simulated collaborative-robot sorting cell: a 6-axis cobot with a vacuum gripper sorts colored cubes
into bins using a YOLO detector trained on synthetic data, a safety layer that blocks unsafe motion, and an
AI operator agent that turns plain-language work orders (Portuguese or English) into safe, validated robot
actions.

> **Simulation only.** Work in progress; the full README lands with milestone M6.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/fetch_assets.py
cobotsort selfcheck
```
