import itertools

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


def run_policy(policy, preprocess, task, fixed_noise):
    raw_batch = {
        "observation.state": torch.zeros(1, 6),
        "observation.images.camera1": torch.zeros(1, 3, 256, 256),
        "observation.images.camera2": torch.zeros(1, 3, 256, 256),
        "observation.images.camera3": torch.zeros(1, 3, 256, 256),
        "task": [task],
    }

    batch = preprocess(raw_batch)

    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(batch, noise=fixed_noise)

    return action_chunk.detach().cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "lerobot/smolvla_base"

    tasks = [
        "pick up the cube",
        "move the object to the left",
        "put the cube into the box",
        "move the object to the right",
    ]

    print("device:", device)
    print("model:", model_id)

    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
    preprocess, _ = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    torch.manual_seed(42)

    fixed_noise = torch.randn(
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        device=device,
    )

    outputs = {}

    for task in tasks:
        action_chunk = run_policy(policy, preprocess, task, fixed_noise)
        outputs[task] = action_chunk
        print("\nTASK:", task)
        print("action_chunk shape:", tuple(action_chunk.shape))
        print("first action:", action_chunk[0, 0])
        print("chunk mean:", action_chunk.mean().item())
        print("chunk std:", action_chunk.std().item())

    print("\nPairwise action_chunk distances:")
    for a, b in itertools.combinations(tasks, 2):
        diff = outputs[a] - outputs[b]
        l2 = torch.norm(diff).item()
        mae = diff.abs().mean().item()
        print(f"{a!r} vs {b!r}: L2={l2:.6f}, MAE={mae:.6f}")


if __name__ == "__main__":
    main()
