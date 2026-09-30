import glob
import numpy as np
import torch

from torch.utils.data import DataLoader
from preprocessing.datasets import MultiFeatureDataset

def make_hailo_calibration_dataset(
    model,
    shards_dir,
    output_path,
    num_samples=1024,
    batch_size=32,
    device="cpu",
):

    print("Generating calibration set")

    shard_paths = sorted(
        glob.glob(f"{shards_dir}/train_unmasked_*.pt")
    )

    if not shard_paths:
        raise RuntimeError(
            f"No train_unmasked_*.pt shards found in {shards_dir}"
        )

    model = model.to(device)
    model.eval()

    samples = []

    with torch.no_grad():
        for shard_path in shard_paths:
            print(f"Loading {shard_path}")

            shard = torch.load(
                shard_path,
                map_location="cpu",
            )

            dataset = MultiFeatureDataset(
                shard["X_train"],
                shard["y_train"],
                shuffle=False,
            )

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
            )

            for X, _ in loader:
                X = X.to(device)

                embeddings = model.embedding_model(X)

                # [B, 336, 32]
                embeddings = (
                    embeddings
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                samples.append(embeddings)

                collected = sum(
                    x.shape[0] for x in samples
                )

                if collected >= num_samples:
                    break

            if collected >= num_samples:
                break

    calibration = np.concatenate(samples, axis=0)
    calibration = calibration[:num_samples]

    # Hailo DFC represents the parsed NWC transformer input internally
    # as NHWC with a singleton height dimension.
    #
    # [N, 336, 32] -> [N, 1, 336, 32]
    calibration = np.expand_dims(calibration, axis=1)

    print("Calibration dataset")
    print("  shape:", calibration.shape)
    print("  dtype:", calibration.dtype)
    print("  min:", calibration.min())
    print("  max:", calibration.max())
    print("  mean:", calibration.mean())
    print("  std:", calibration.std())

    flat = calibration.reshape(-1)

    for p in [0, 0.1, 1, 50, 99, 99.9, 100]:
        print(
            f"  p{p:5.1f}: "
            f"{np.percentile(flat, p):.6f}"
        )

    np.save(output_path, calibration)

    print(f"Saved to {output_path}")
