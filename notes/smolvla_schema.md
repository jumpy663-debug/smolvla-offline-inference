# SmolVLA Schema Notes

## Model

```text
model_id = lerobot/smolvla_base
```

## Input Features

```text
observation.state: shape=(6,)
observation.images.camera1: shape=(3, 256, 256)
observation.images.camera2: shape=(3, 256, 256)
observation.images.camera3: shape=(3, 256, 256)
```

Language is added by the preprocessor:

```text
observation.language.tokens: shape=(1, 48)
observation.language.attention_mask: shape=(1, 48)
```

## Output Features

```text
action: shape=(6,)
```

## Action Chunk

```text
chunk_size = 50
n_action_steps = 50
predict_action_chunk output = (batch_size, 50, 6)
select_action output = (batch_size, 6)
```

## LIBERO Mismatch

`lerobot/libero` has:

```text
observation.images.image
observation.images.image2
observation.state: shape=(8,)
action: shape=(7,)
```

`smolvla_base` expects:

```text
observation.images.camera1
observation.images.camera2
observation.images.camera3
observation.state: shape=(6,)
action: shape=(6,)
```

Temporary adapter:

```text
image  -> camera1
image2 -> camera2
image2 -> camera3
state[:6] -> state
```

This is only for offline data-flow verification, not for formal LIBERO evaluation.
