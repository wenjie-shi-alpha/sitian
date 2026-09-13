# Sitian

Agent RL for city-level air-quality forecasting. Agents gather observations and model guidance, submit structured forecasts, and learn from verifiable outcome rewards.

## Core design

- **Real forecasting tasks:** each episode represents a forecast issuance, with multi-turn tools for observations, weather conditions, model guidance, and historical cases.
- **Verifiable rewards:** observations score forecast outcomes; tool outputs ground evidence claims. Training rewards and forecast metrics remain separate, with constraints on wide intervals and citation gaming.
- **Multi-turn RL:** veRL / vLLM integration with on-policy rollouts, Dr.GRPO, DAPO, and temporal and spatial holdout evaluation.

## Quick start

Requires Python 3.11+. This synthetic example needs no GPU, model server, or operational data:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
fh gen --out /tmp/sitian-demo
fh baselines --cases '/tmp/sitian-demo/*'
```

The example checks the environment and scorer. Real cases, model weights, and training data require separate preparation. Data availability assumptions and training status are documented below.

## Documentation

- [Research question](docs/SCIENTIFIC_CLAIM.md) (Chinese)
- [Training design](docs/TRAINING_DESIGN.md) (Chinese)
- [Data admission and limitations](docs/TRAINING_READY_2026-09-12.md) (Chinese)
- [Development and GPU setup](docs/LOCAL_SETUP.md) (Chinese)

Core code lives in [`src/sitian/`](src/sitian/), with training adapters in [`integrations/`](src/sitian/integrations/).
