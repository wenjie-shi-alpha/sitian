# 司天 Sitian

面向城市空气质量预报的 Agent RL 研究环境。智能体自主读取观测与模式证据、提交结构化预报，并从可验证的结果奖励中学习。

## 核心设计

- **真实决策环境**：一个 episode 对应一个起报时次，支持实况、气象形势、模式指导与历史案例等多轮工具调用。
- **可验证奖励**：以观测真值评价预报，以工具返回核验证据；区分训练奖励与预测结果，约束宽区间和引用刷分。
- **多轮强化学习**：接入 veRL / vLLM 的同策略 rollout、Dr.GRPO 与 DAPO，配套时间和空间留出评估。

## 快速体验

需要 Python 3.11+。以下合成案例无需 GPU、模型服务或真实业务数据：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
fh gen --out /tmp/sitian-demo
fh baselines --cases '/tmp/sitian-demo/*'
```

输出用于检查任务环境和评分器。真实案例、模型权重与训练数据需另行准备；供数时间假设、数据准入和训练状态见下方文档。

## 文档

- [任务与研究问题](docs/SCIENTIFIC_CLAIM.md)
- [训练设计](docs/TRAINING_DESIGN.md)
- [数据准入与已知限制](docs/TRAINING_READY_2026-09-12.md)
- [开发与 GPU 环境](docs/LOCAL_SETUP.md)

核心实现位于 [`src/sitian/`](src/sitian/)；多轮训练适配位于 [`integrations/`](src/sitian/integrations/)。
