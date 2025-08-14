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

    base_path = args.orig_base_path
    new_ds_path = args.new_base_path
    train_path = os.path.join(base_path, "train")
    eval_path = os.path.join(base_path, "eval")

    def add_episode_ids_to_dataset(is_train_ds):
        if is_train_ds:
            orig_papth = train_path
            subdirname = "train"
        else:
            orig_papth = eval_path
            subdirname = "eval"
            
        os.makedirs(os.path.join(new_ds_path, f"{subdirname}"), exist_ok=True)
            
        epi_id = 0
        for dirpath, dirnames, filenames in tqdm(os.walk(orig_papth)):
            for filename in filenames:
                if filename.split(".")[-1] == "parquet":
                    fpath = os.path.join(dirpath, filename)
                    df = pd.read_parquet(fpath)
                    df["episode_id"] = epi_id
                    table = pa.Table.from_pandas(df)
                    to_path = os.path.join(new_ds_path, f"{subdirname}", f"{epi_id}.parquet")
                    pq.write_table(table, to_path, compression='zstd')
                    epi_id += 1

    add_episode_ids_to_dataset(is_train_ds=True)
    add_episode_ids_to_dataset(is_train_ds=False)