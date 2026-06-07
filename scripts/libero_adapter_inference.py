import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
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
    dataset_id = "lerobot/libero"

    print("device:", device)
    policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    dataset = LeRobotDataset(dataset_id)
    frame = dict(dataset[0])

    print("dataset:", dataset_id)
    print("dataset length:", len(dataset))
    print("task:", frame["task"])
    print("original state shape:", tuple(frame["observation.state"].shape))
    print("original action shape:", tuple(frame["action"].shape))

    adapted = {
        "observation.images.camera1": frame["observation.images.image"],
        "observation.images.camera2": frame["observation.images.image2"],
        "observation.images.camera3": frame["observation.images.image2"],
        "observation.state": frame["observation.state"][:6],
        "task": frame["task"],
    }

    print("\nadapted raw frame:")
    describe_batch(adapted)

    batch = preprocess(adapted)

    print("\nprocessed batch:")
    describe_batch(batch)

    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(batch)
        single_action = policy.select_action(batch)
        single_action = postprocess(single_action)

    print("\naction_chunk shape:", tuple(action_chunk.shape))
    print("single_action shape:", tuple(single_action.shape))
    print("single_action:", single_action.detach().cpu())


if __name__ == "__main__":
    main()
