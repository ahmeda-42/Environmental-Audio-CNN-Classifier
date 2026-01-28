import os
import pandas as pd


def find_dataset_root(dataset_root):

    print(f"finding dataset root in: {dataset_root}")
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(
            f"dataset root not found: {dataset_root}. "
            "Make sure the UrbanSound8K folder is downloaded and extracted."
        )

    direct_meta = os.path.join(dataset_root, "metadata", "UrbanSound8K.csv")
    if os.path.exists(direct_meta):
        return dataset_root

    nested_root = os.path.join(dataset_root, "UrbanSound8K")
    nested_meta = os.path.join(nested_root, "metadata", "UrbanSound8K.csv")
    if os.path.exists(nested_meta):
        return nested_root

    for entry in os.listdir(dataset_root):
        candidate = os.path.join(dataset_root, entry, "metadata", "UrbanSound8K.csv")
        if os.path.exists(candidate):
            return os.path.join(dataset_root, entry)

    raise FileNotFoundError(
        "metadata not found. Expected UrbanSound8K.csv under "
        f"{dataset_root}/metadata or a nested UrbanSound8K folder."
    )


def build_csv(dataset_root, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    dataset_root = find_dataset_root(dataset_root)
    metadata_path = os.path.join(dataset_root, "metadata", "UrbanSound8K.csv")

    meta = pd.read_csv(metadata_path)
    rows = []
    for _, row in meta.iterrows():
        fold = f"fold{row['fold']}"
        filename = row["slice_file_name"]
        file_path = os.path.join(dataset_root, "audio", fold, filename)
        rows.append(
            {
                "file_path": file_path,
                "label": row["class"],
                "fold": int(row["fold"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"wrote {output_csv}")