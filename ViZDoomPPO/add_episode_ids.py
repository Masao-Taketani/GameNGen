import argparse
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training RL agent and data collection program")
    parser.add_argument(
        "--orig_base_path",
        type=str,
        help="Specify the original base path that includes collected training and eval data.",
    )
    parser.add_argument(
        "--new_base_path",
        type=str,
        help="Specify new base path that is supposed to have collected training and eval data with episode ids.",
    )
    args = parser.parse_args()

    base_path, new_ds_path = args.orig_base_path, args.new_base_path

    def get_total_length(orig_papth):
        total = 0
        for dirpath, dirnames, filenames in tqdm(os.walk(orig_papth)): total += 1
        return total

    def add_episode_ids_to_dataset(is_train_ds):
        subdirname = "train" if is_train_ds else "eval"
        orig_papth = os.path.join(base_path, subdirname)
        os.makedirs(os.path.join(new_ds_path, f"{subdirname}"), exist_ok=True)
            
        epi_id = 0
        total = get_total_length(orig_papth)
        for dirpath, dirnames, filenames in tqdm(os.walk(orig_papth), total=total, desc=f"Outer Loop({subdirname})"):
            for filename in tqdm(filenames, desc="Inner Loop", leave=False):
                if filename.split(".")[-1] == "parquet":
                    fpath = os.path.join(dirpath, filename)
                    df = pd.read_parquet(fpath)
                    df["episode_id"] = epi_id
                    table = pa.Table.from_pandas(df)
                    to_path = os.path.join(new_ds_path, f"{subdirname}", f"episode_{epi_id}.parquet")
                    pq.write_table(table, to_path, compression='zstd')
                    epi_id += 1

    add_episode_ids_to_dataset(is_train_ds=True)
    add_episode_ids_to_dataset(is_train_ds=False)