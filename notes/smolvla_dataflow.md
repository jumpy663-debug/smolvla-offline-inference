# SmolVLA Data Flow Notes

本文用于梳理 `lerobot/smolvla_base` 的内部数据流。目标不是逐行背源码，而是能在面试中讲清楚：

- 为什么 SmolVLA 是 VLA；
- 图像、语言、state 如何进入模型；
- action chunk 如何生成；
- flow matching 在动作生成里做了什么；
- SmolVLA 和 ACT 的区别。

## 1. Raw Input

SmolVLA 的原始输入由三部分组成：

```text
图像 observation.images.camera1/2/3
语言 task
机器人状态 observation.state
```

在 `lerobot/smolvla_base` 中，模型期望的 schema 是：

```text
observation.images.camera1: (3, 256, 256)
observation.images.camera2: (3, 256, 256)
observation.images.camera3: (3, 256, 256)
observation.state: (6,)
action: (6,)
```

批量化后，输入通常变成：

```text
observation.images.camera*: (B, 3, 256, 256)
observation.state: (B, 6)
task: list[str]
```

## 2. Preprocess

`make_pre_post_processors()` 会构造预处理和后处理模块。

预处理主要做三件事：

```text
图像 -> tensor / device / value range
state -> normalization / device
task text -> language tokens + attention mask
```

预处理后，batch 中会出现：

```text
observation.language.tokens: (B, 48)
observation.language.attention_mask: (B, 48)
```

因此，传入 SmolVLA policy 的不是原始字符串，而是已经 tokenized 的语言输入。

## 3. Policy-Level Flow

SmolVLA policy 的推理入口有两个：

```python
policy.predict_action_chunk(batch)
policy.select_action(batch)
```

区别：

```text
predict_action_chunk:
一次生成完整 action chunk，shape = (B, 50, 6)

select_action:
在线执行接口。内部先生成 action chunk，再把动作放入队列，每次弹出一个 action。
```

在 `SmolVLAPolicy._get_action_chunk()` 中，核心流程是：

```text
prepare_images(batch)
prepare_state(batch)
读取 language tokens / attention mask
model.sample_actions(...)
截断到原始 action_dim
```

## 4. Image Flow

`prepare_images()` 会读取 config 中的 image features：

```text
observation.images.camera1
observation.images.camera2
observation.images.camera3
```

每路图像会经过：

```text
resize_with_pad
[0, 1] -> [-1, 1]
image mask 构造
```

之后图像会进入 VLM 的视觉编码器，得到 image embeddings。

在概念上：

```text
camera images -> vision encoder -> image tokens / embeddings
```

## 5. State Flow

`prepare_state()` 会取当前 state，并 pad 到 `max_state_dim`：

```python
state = pad_vector(state, self.config.max_state_dim)
```

对于 `smolvla_base`：

```text
原始 state_dim = 6
max_state_dim = 32
```

也就是说，6 维 state 会被补零到 32 维。

随后：

```python
state_emb = self.state_proj(state)
```

`state_proj` 把机器人 state 投影到 VLM hidden size，使 state 可以作为 token/embedding 参与后续 Transformer 计算。

## 6. Language Flow

语言指令由 preprocessor 转成：

```text
observation.language.tokens
observation.language.attention_mask
```

在 `embed_prefix()` 中：

```python
lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
```

语言 token 会被 VLM 的 embedding layer 转成 language embeddings。

概念上：

```text
task string -> tokenizer -> token ids -> language embeddings
```

## 7. Prefix: Image + Language + State

SmolVLA 使用 `embed_prefix()` 构造上下文 prefix。

prefix 包括：

```text
image embeddings
language embeddings
state embedding
```

可以理解为：

```text
prefix = 当前世界状态 + 任务语义 + 机器人自身状态
```

这个 prefix 负责告诉模型：

```text
现在看到了什么？
任务要求是什么？
机器人当前在哪里？
```

## 8. Suffix: Noisy Action + Timestep

SmolVLA 不是直接输出 action。它使用 flow matching 生成动作。

在训练和推理中，action side 会构造 suffix：

```text
noisy actions
timestep
```

`embed_suffix()` 做的事：

```text
noisy action -> action_in_proj
timestep -> sinusoidal embedding
concat(action_emb, time_emb)
MLP fusion
```

概念上：

```text
suffix = 当前待去噪的动作序列 + 当前去噪时间
```

## 9. Flow Matching Training

训练时模型看到真实 action chunk。

代码中的核心公式：

```python
noise = sample_noise(actions.shape)
time = sample_time(batch_size)

x_t = time * noise + (1 - time) * actions
u_t = noise - actions
```

含义：

```text
actions: 专家动作
noise: 随机噪声动作
x_t: 专家动作和噪声之间的中间状态
u_t: 从专家动作指向噪声的速度目标
```

模型输入：

```text
prefix: image + language + state
suffix: x_t + time
```

模型输出：

```text
v_t = predicted velocity
```

loss：

```python
MSE(u_t, v_t)
```

直观理解：

```text
模型学习在给定图像、语言和 state 的条件下，
如何描述 action 分布和 noise 之间的流动方向。
```

## 10. Flow Matching Inference

推理时没有真实 action。

所以从随机 noise 开始：

```text
x_t = random noise action chunk
```

然后经过多步迭代：

```text
prefix = image + language + state
suffix = current noisy action + timestep
model predicts velocity
update action chunk
```

最终得到：

```text
action_chunk: (B, 50, 6)
```

这就是 SmolVLA 的动作生成过程：

```text
noise action chunk -> denoising / flow steps -> executable action chunk
```

## 11. VLM + Action Expert

SmolVLA 不是让 VLM 直接输出文字，也不是让 VLM 直接输出 action。

结构上可以理解为：

```text
VLM:
理解图像和语言上下文

Action Expert:
在 VLM 上下文条件下生成连续动作
```

代码中相关模块：

```python
SmolVLMWithExpertModel
state_proj
action_in_proj
action_out_proj
action_time_mlp_in
action_time_mlp_out
```

作用：

```text
state_proj: state -> VLM hidden size
action_in_proj: action -> expert hidden size
action_out_proj: expert hidden -> action dim
action_time_mlp: 融合 noisy action 和 timestep
```

## 12. select_action Queue

`predict_action_chunk()` 一次输出：

```text
(B, 50, 6)
```

而真实机器人执行通常一次只执行一步动作。

所以 `select_action()` 使用 action queue：

```text
如果 queue 为空：
    生成新的 action chunk
    把 50 个 action 放进 queue
每次调用：
    pop 一个 action
```

这使 SmolVLA 可以像在线策略一样被调用。

## 13. SmolVLA vs ACT

| 维度 | ACT | SmolVLA |
|---|---|---|
| 输入 | 图像 + state | 图像 + state + language |
| 输出 | action chunk | action chunk |
| 生成方式 | CVAE + Transformer | VLM + Action Expert + Flow Matching |
| latent / noise | CVAE latent | flow matching noise |
| 语言能力 | 通常没有 | 有语言条件 |
| 项目定位 | 低层模仿学习策略 | 视觉-语言-动作模型 |

一句话对比：

```text
ACT 学习从视觉和状态到动作 chunk 的模仿策略；
SmolVLA 在视觉、语言和状态条件下，通过 flow matching 生成连续动作 chunk。
```

## 14. 面试口述版

SmolVLA 的输入包括多视角图像、语言指令和机器人 state。预处理会把语言转成 tokens，把图像和 state 放到模型期望的格式中。模型内部先通过 VLM 编码图像和语言，并通过 state projection 把机器人状态变成可融合的 embedding，这些构成 prefix。动作生成部分不是直接回归 action，而是从 noisy action chunk 开始，结合 timestep 构造 suffix，通过 action expert 做 flow matching 去噪，最终生成 50 步连续动作。`predict_action_chunk` 会输出完整动作序列，`select_action` 则通过队列每次取出一步用于在线执行。
