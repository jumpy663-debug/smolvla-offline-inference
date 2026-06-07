# SmolVLA 离线推理与数据流分析

本项目基于 Hugging Face LeRobot 的 `lerobot/smolvla_base`，验证轻量 VLA 模型在本地环境中的加载、输入 schema、语言 token 预处理和离线 action chunk 推理流程。

项目重点不是机器人真机部署，也不是 LIBERO 正式评测，而是回答一个更基础的问题：

```text
SmolVLA 如何接收图像、语言指令和机器人状态，并输出连续动作序列？
```

## 当前结论

- `lerobot/smolvla_base` 可在 RTX 4060 Laptop GPU 上加载并运行推理。
- 模型期望 3 路视觉输入、6 维 state、6 维 action。
- 模型一次输出 50 步 action chunk，shape 为 `(batch_size, 50, 6)`。
- 已完成 synthetic batch 推理和真实 `lerobot/libero` frame 的 schema adapter 推理。
- `lerobot/libero` 与 `smolvla_base` 存在 schema mismatch，因此当前结果只能作为离线推理适配实验，不能作为正式策略评测。

## 环境

- OS: WSL2 Ubuntu
- IDE: VS Code Remote WSL
- Environment: conda
- GPU: NVIDIA RTX 4060 Laptop GPU, 8GB
- PyTorch: `2.11.0+cu130`
- LeRobot commit: `b8ad81bf`
- Model: `lerobot/smolvla_base`
- Dataset used for real-frame adapter test: `lerobot/libero`

## SmolVLA Base Schema

模型加载后得到的关键配置：

```text
chunk_size: 50
n_action_steps: 50
max_state_dim: 32
max_action_dim: 32
```

输入 feature：

```text
observation.state: shape=(6,)
observation.images.camera1: shape=(3, 256, 256)
observation.images.camera2: shape=(3, 256, 256)
observation.images.camera3: shape=(3, 256, 256)
```

输出 feature：

```text
action: shape=(6,)
```

## 实验 1：Synthetic Offline Inference

脚本：

```bash
python scripts/synthetic_inference.py
```

构造输入：

```text
observation.state: zeros, shape=(1, 6)
observation.images.camera1/2/3: zeros, shape=(1, 3, 256, 256)
task: "pick up the cube"
```

关键输出：

```text
observation.language.tokens: shape=(1, 48)
observation.language.attention_mask: shape=(1, 48)
action_chunk: shape=(1, 50, 6)
single action: shape=(1, 6)
```

意义：验证 `language + vision + state -> SmolVLA -> action chunk` 的基本数据流。

## 实验 2：LIBERO Frame Adapter Inference

脚本：

```bash
python scripts/libero_adapter_inference.py
```

`lerobot/libero` 的原始 schema：

```text
observation.images.image
observation.images.image2
observation.state: shape=(8,)
action: shape=(7,)
task: language instruction
```

`smolvla_base` 期望：

```text
observation.images.camera1
observation.images.camera2
observation.images.camera3
observation.state: shape=(6,)
action: shape=(6,)
```

当前 adapter：

```text
observation.images.image  -> observation.images.camera1
observation.images.image2 -> observation.images.camera2
observation.images.image2 -> observation.images.camera3
observation.state[:6]     -> observation.state
```

已验证输出：

```text
action_chunk shape: (1, 50, 6)
single action shape: (1, 6)
```

限制：该 adapter 仅用于验证离线推理链路。由于 state/action 维度和相机定义不完全匹配，不能据此声称 SmolVLA 在 LIBERO 上完成有效策略执行。

## 实验 3：语言指令敏感性分析

为排除 flow matching 随机噪声的影响，实验固定同一份 noise、同一组 zero image 和 zero state，只改变语言指令，比较输出 action chunk 的差异。

| 指令对比 | L2 Distance | MAE |
|---|---:|---:|
| pick up the cube vs move the object to the left | 3.489 | 0.145 |
| pick up the cube vs put the cube into the box | 2.120 | 0.108 |
| pick up the cube vs move the object to the right | 2.383 | 0.104 |
| move the object to the left vs put the cube into the box | 4.234 | 0.162 |
| move the object to the left vs move the object to the right | 4.207 | 0.158 |
| put the cube into the box vs move the object to the right | 1.095 | 0.054 |

结论：在固定 noise 的条件下，不同语言指令仍会导致 action chunk 变化，说明 language tokens 会参与 SmolVLA 的动作生成。但由于当前使用 synthetic zero image/state，该实验不能证明动作语义正确，只能证明语言条件会影响输出。

## 技术理解

SmolVLA 与 ACT 的核心差异：

```text
ACT:
image + state + latent -> Transformer -> action chunk

SmolVLA:
image + language + state + noisy actions
-> VLM prefix + action expert suffix
-> flow matching denoising
-> action chunk
```

SmolVLA 推理流程：

```text
raw frame
-> preprocess
-> image tensors + state + language tokens
-> predict_action_chunk
-> 50-step action chunk
-> select_action
-> single executable action
```

## 当前局限

- 当前只完成离线推理，没有进行真实机器人执行。
- `lerobot/libero` 和 `smolvla_base` 存在 schema mismatch。
- 真实策略质量需要匹配训练 schema 的数据集或真机环境才能评估。
- 8GB 显存适合推理和小规模实验，不适合盲目全量微调大规模 VLA。

## 后续方向

- 寻找与 `smolvla_base` schema 完全匹配的 SO100 数据集。
- 对比 `select_action` 与 `predict_action_chunk` 的输出行为。
- 分析语言指令变化对 action chunk 的影响。
- 在小规模数据上尝试轻量 fine-tuning 或只训练 action expert。

## 面试讲解摘要

可以将本项目概括为：

> 我基于 LeRobot 加载了 `lerobot/smolvla_base`，分析其输入输出 schema，并完成了 synthetic batch 和真实 `lerobot/libero` frame 的离线推理实验。SmolVLA base 期望 3 路相机、6 维 state 和语言指令输入，输出 50 步、6 维 action chunk。由于 LIBERO 数据集的 state/action 维度和 camera keys 与模型不完全匹配，我实现了一个临时 schema adapter 来验证数据流，但没有将其包装成正式评测结果。这个项目重点展示我对 VLA 数据流、语言条件动作生成和模型/数据 schema 对齐问题的理解。
