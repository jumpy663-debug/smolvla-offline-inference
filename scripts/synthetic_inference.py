import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


def describe_batch(batch):
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(key, tuple(value.shape), value.dtype, value.device)
        else:
            print(key, type(value), value if isinstance(value, (str, int, float, bool, list)) else "")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "lerobot/smolvla_base"

    print("device:", device)
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    raw_batch = {
        "observation.state": torch.zeros(1, 6),
        "observation.images.camera1": torch.zeros(1, 3, 256, 256),
        "observation.images.camera2": torch.zeros(1, 3, 256, 256),
        "observation.images.camera3": torch.zeros(1, 3, 256, 256),
        "task": ["pick up the cube"],
    }

    print("\nraw batch:")
    describe_batch(raw_batch)

    batch = preprocess(raw_batch)

    print("\nprocessed batch:")
    describe_batch(batch)

    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(batch)
        single_action = policy.select_action(batch)
        single_action = postprocess(single_action)

    print("\naction_chunk shape:", tuple(action_chunk.shape))
    print("action_chunk dtype:", action_chunk.dtype)
    print("action_chunk device:", action_chunk.device)
    print("single_action shape:", tuple(single_action.shape))
    print("single_action:", single_action.detach().cpu())


if __name__ == "__main__":
    main()
