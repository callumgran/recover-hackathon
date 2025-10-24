import os
import random
import sys
import typing
from pathlib import Path
from urllib.error import URLError

import numpy as np
import polars as pl
import torch
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.base import BaseDataset, index_encode_str, index_encode_str_batch


class WorkOperationsDataset(BaseDataset):
    competition = "hackathon-recover-x-cogito"
    resources: typing.ClassVar[list[str]] = [
        "train.csv",
        "val.csv",
        "test.csv",
        "tickets.csv",
    ]

    __rooms: typing.ClassVar[list[str]] = [
        "andre områder",
        "kjøkken",
        "stue",
        "gang",
        "soverom",
        "bad",
        "bod",
        "vaskerom",
        "wc",
        "kjeller",
        "garasje",
    ]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        download: bool = False,
        sampling_strategy: list[dict] | None = None,
        seed: int | None = None,
        np_rng: np.random.Generator | None = None,
        py_rng: random.Random | None = None,
        num_clusters: int = 388,
    ):
        super().__init__(root)
        self.split = split
        self.seed = seed
        self.num_clusters = num_clusters

        self.np_rng = np_rng if np_rng is not None else np.random.default_rng(seed)
        self.py_rng = py_rng if py_rng is not None else random.Random(seed)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.data = self._load_data()
        self.tickets = self._load_tickets()
        self.sampling_strategy = sampling_strategy or [
            {
                "subset_size": 0.5,
                "sample_pct": 0.5,
                "use_balanced_data": True,
                "use_sampled_calculus": True,
            },
            {
                "subset_size": 0.5,
                "sample_pct": 0.3,
                "use_balanced_data": False,
                "use_sampled_calculus": True,
            },
        ]
        assert abs(sum(cfg["subset_size"] for cfg in self.sampling_strategy) - 1.0) < 1e-6, (
            "Subset sizes must sum to 1."
        )
        if split == "test":
            self.sampling_strategy = [
                {
                    "subset_size": 1.0,
                    "sample_pct": 0.0,
                    "use_balanced_data": False,
                    "use_sampled_calculus": False,
                }
            ]

        self.shuffle()

    def set_sampling_strategy(self, strategy: list[dict]):
        assert abs(sum(cfg["subset_size"] for cfg in strategy) - 1.0) < 1e-6, "Subset sizes must sum to 1."
        self.sampling_strategy = strategy

    def shuffle(self):
        self._sample()
        self.col_idx = {name: i for i, name in enumerate(self.data.columns)}
        return self

    def __len__(self):
        return self.data.height

    def __getitem__(self, idx):
        row = self.data.row(idx)
        return {
            "id": row[self.col_idx["id"]],
            "X": torch.tensor(self._index_encode(row[self.col_idx["X"]])),
            "Y": torch.tensor(self._index_encode(row[self.col_idx["Y"]])),
            "project_id": row[self.col_idx["project_id"]],
            "room_cluster": row[self.col_idx["room_cluster"]],
            "room_cluster_one_hot": torch.tensor(row[self.col_idx["room_cluster_one_hot"]]),
            "calculus": row[self.col_idx["calculus"]],
            "X_codes": torch.tensor(row[self.col_idx["X"]]),
            "Y_codes": torch.tensor(row[self.col_idx["Y"]]),
        }

    def to_cluster_names(self, it: typing.Iterable) -> list[str]:
        """Convert a list of index encoded wo codes or a list of wo cluster codes to their corresponding names."""
        if not isinstance(it, (list | np.ndarray | torch.Tensor)):
            it = list(it)
        if isinstance(it, torch.Tensor):
            if it.ndim > 1:
                it = it.squeeze()
        if isinstance(it, (np.ndarray | torch.Tensor)):
            it = it.tolist()
        if self._is_index(it):
            it = self.to_cluster_codes(it)

        if not isinstance(it, list):
            raise TypeError("Input must be a list, numpy array, or torch tensor.")

        return [self.code_to_wo.get(code, "Unknown") for code in it]

    def _cluster_room(self, room_name: str) -> str:
        if room_name.lower() in self.__rooms:
            return room_name.lower()
        for room in self.__rooms:
            if room in room_name.lower():
                return room
        return "ukjent"

    def _index_encode(self, label_list: list[int]) -> list[int]:
        vec = np.zeros(self.num_clusters, dtype=np.int8)
        for i in label_list:
            vec[i] = 1
        return vec.tolist()

    def _index_encode_batch(self, labels: list[list[int]]) -> np.ndarray:
        batch_vec = np.zeros((len(labels), self.num_clusters), dtype=np.int8)
        for i, label_list in enumerate(labels):
            for j in label_list:
                batch_vec[i, j] = 1
        return batch_vec

    def _load_data(self):
        lf = pl.scan_csv(os.path.join(self.data_folder, f"{self.split}.csv"))
        data = (
            lf.with_columns(
                pl.col("room").map_elements(self._cluster_room, return_dtype=pl.Utf8).alias("room_cluster"),
            )
            .group_by(["project_id", "room", "room_cluster"])
            .agg(
                pl.col("id").first(),
                pl.col("work_operation_cluster_name").alias("wo_names"),
                pl.col("work_operation_cluster_code").alias("wo_codes"),
            )
            .with_columns(
                pl.col("wo_codes")
                .map_elements(lambda x: self._index_encode(x), return_dtype=pl.List(pl.Int8))
                .alias("wo_codes_index_encoded"),
                pl.col("room_cluster")
                .map_elements(
                    lambda x: index_encode_str(x, len(self.__rooms), self.room_to_index),
                    return_dtype=pl.List(pl.Int8),
                )
                .alias("room_cluster_one_hot"),
            )
            .filter(pl.col("wo_codes").list.len() > 1)
            .sort(["project_id", "room", "wo_codes"])
            .with_row_index()
            .sort("index")
            .collect()
        )
        return data

    def _load_tickets(self):
        df = pl.scan_csv(os.path.join(self.data_folder, "tickets.csv")).collect()
        tickets = df.with_columns(
            (pl.col("n_tickets").cast(pl.Float64) / df.select(pl.col("n_tickets").max()).item()).alias(
                "normalized_n_tickets"
            )
        )
        return tickets

    def _check_exists(self) -> bool:
        return all(os.path.exists(os.path.join(self.data_folder, filename)) for filename in self.resources)

    @property
    def data_folder(self) -> str:
        return self.root

    @property
    def wo_to_code(self) -> dict[str, int]:
        return dict(
            self.data.select(pl.col("wo_names").explode(), pl.col("wo_codes").explode())
            .unique()
            .sort("wo_codes")
            .iter_rows(named=False)
        )

    @property
    def code_to_wo(self) -> dict[int, str]:
        return dict(
            self.data.select(pl.col("wo_codes").explode(), pl.col("wo_names").explode())
            .unique()
            .sort("wo_codes")
            .iter_rows(named=False)
        )

    @property
    def room_to_index(self) -> dict[str, int]:
        return {room: idx for idx, room in enumerate(self.__rooms)}

    @property
    def index_to_room(self) -> dict[int, str]:
        return dict(enumerate(self.__rooms))

    def _sample(self):
        ids = (
            self.data.select(pl.col("project_id"))
            .unique(maintain_order=True)
            .sample(fraction=1.0, shuffle=True, seed=self.py_rng.randint(0, 1_000_000))
            .to_numpy()
            .flatten()
        )
        subsets = []

        j = 0
        for i, strategy in enumerate(self.sampling_strategy):
            subset_size = strategy["subset_size"]
            n_ids = int(len(ids) * subset_size)
            if i == len(self.sampling_strategy) - 1:
                n_ids = len(ids) - j
            subset_ids = ids[j : j + n_ids]
            j += n_ids

            subset = self.data.filter(pl.col("project_id").is_in(subset_ids))

            df = subset.with_columns(
                pl.Series(
                    "n_samples",
                    self.np_rng.integers(
                        0,
                        (
                            strategy["sample_pct"] * subset.select(pl.col("wo_codes").list.len()).to_numpy().flatten()
                        ).astype(int)
                        + 1,
                    ),
                    dtype=pl.Int64,
                )
            )

            if strategy["use_balanced_data"]:
                exploded = (
                    df.explode("wo_codes")
                    .select(["index", "wo_codes"])
                    .join(
                        self.tickets,
                        left_on="wo_codes",
                        right_on="work_operation_cluster_code",
                        how="left",
                    )
                    .fill_nan(0)
                )
            else:
                exploded = (
                    df.explode("wo_codes")
                    .select(["index", "wo_codes"])
                    .with_columns(pl.lit(1).alias("normalized_n_tickets"))
                    .fill_nan(0)
                )
            exploded = (
                exploded.with_columns(
                    pl.Series(
                        "random_weight",
                        self.np_rng.uniform(0, 1, len(exploded)),
                        dtype=pl.Float64,
                    )
                )
                .with_columns((pl.col("random_weight") * pl.col("normalized_n_tickets")).alias("weight"))
                .sort(by=["index", "weight"], descending=[False, True])
            )

            df = (
                exploded.group_by("index")
                .agg("wo_codes")
                .join(df.select(["index", "n_samples"]), on="index", how="inner")
                .with_columns(
                    pl.col("wo_codes").list.slice(0, pl.col("n_samples")).alias("removed_codes"),
                    pl.col("wo_codes")
                    .list.slice(
                        pl.col("n_samples"),
                        pl.col("wo_codes").list.len() - pl.col("n_samples"),
                    )
                    .alias("remaining_codes"),
                )
            )

            keep_cols = [c for c in subset.columns if c not in ["X", "Y"]]
            base_df = subset.select(keep_cols)
            base_df = base_df.join(
                df.select(["index", "remaining_codes", "removed_codes"]),
                on="index",
                how="inner",
            )
            base_df = base_df.rename({"remaining_codes": "X", "removed_codes": "Y"})
            base_df = base_df.with_columns(
                pl.lit(strategy["use_balanced_data"]).alias("use_balanced_data"),
                pl.lit(strategy["sample_pct"]).alias("sample_pct"),
                pl.lit(strategy["subset_size"]).alias("subset_size"),
                pl.lit(strategy["use_sampled_calculus"]).alias("use_sampled_calculus"),
            )

            if strategy["use_sampled_calculus"]:
                right = base_df.select(["project_id", "room", "X", "room_cluster"]).rename({"X": "sampled_X"})
            else:
                right = base_df.select(["project_id", "room", "wo_codes", "room_cluster"]).rename(
                    {"wo_codes": "sampled_X"}
                )
            left = base_df.select(["project_id", "room"]).with_row_index("row_index")
            joined = left.join(right, on="project_id", how="left")
            joined = joined.filter(pl.col("room") != pl.col("room_right"))
            calculus_df = (
                joined.group_by("row_index")
                .agg(
                    pl.struct(
                        work_operations="sampled_X",
                        room="room_right",
                        room_cluster="room_cluster",
                        schema={
                            "work_operations": pl.List(pl.Int32),
                            "room": pl.Utf8,
                            "room_cluster": pl.Utf8,
                        },
                    ).alias("calculus")
                )
                .sort("row_index")
            )
            if "calculus" in base_df.columns:
                base_df = base_df.drop("calculus")
            full_calculus = [None] * subset.height
            for idx, val in zip(calculus_df["row_index"], calculus_df["calculus"].to_list(), strict=False):
                full_calculus[idx] = val
            full_calculus = [x if x is not None else [] for x in full_calculus]

            calculus_index = []
            for calc_list in full_calculus:
                new_calc_list = []
                for entry in calc_list:
                    entry = dict(entry)
                    entry["work_operations_index_encoded"] = torch.tensor(self._index_encode(entry["work_operations"]))
                    entry["room_cluster_one_hot"] = torch.tensor(
                        index_encode_str(entry["room_cluster"], len(self.__rooms), self.room_to_index)
                    )
                    new_calc_list.append(entry)
                calculus_index.append(new_calc_list)

            calculus_df = pl.DataFrame({"calculus": calculus_index}, strict=False)
            base_df = base_df.with_columns(calculus_df)

            subsets.append(base_df)

        self.data = (pl.concat(subsets)).drop("index").with_row_index().sort("index")
        self._exploded_dataframe = (
            self.data.select(
                [
                    "index",
                    "project_id",
                    "room",
                    "room_cluster",
                    "wo_codes",
                    "Y",
                    "use_balanced_data",
                    "sample_pct",
                    "subset_size",
                    "use_sampled_calculus",
                ]
            )
            .explode(["wo_codes"])
            .with_columns(pl.col("Y").list.contains(pl.col("wo_codes")).alias("is_hidden"))
            .rename({"wo_codes": "work_operation"})
            .drop("Y")
            .unique(["project_id", "room", "work_operation"])
        )

    def _is_index(self, it: typing.Iterable) -> bool:
        """Check if the iterable is a one-hot or index encoded vector."""
        return isinstance(it, (list | np.ndarray | torch.Tensor)) and all(
            isinstance(x, int) and (x == 0 or x == 1) for x in it
        )

    def get_exploded_dataframe(self):
        return self._exploded_dataframe

    def to_cluster_codes(self, it: typing.Iterable) -> list[int]:
        """Convert a iterable of index encoded vectors to a list of indices."""
        return [i + 1 for i, val in enumerate(it) if val == 1]

    def extra_repr(self) -> str:
        sampling_strat = "\n".join(
            f"\t - Subset size: {cfg['subset_size']}, Sample percentage: {cfg['sample_pct']}, "
            f"Use weighted sampling: {cfg['use_balanced_data']}, "
            f"Use sampled calculus: {cfg['use_sampled_calculus']}"
            for cfg in self.sampling_strategy
        )
        return f"Split: {self.split}, \nNumber of rooms: {len(self.__rooms)}\nSampling strategy:\n{sampling_strat}"


if __name__ == "__main__":
    dataset = WorkOperationsDataset(root="data", split="train", download=True)
    print(dataset)
    print(dataset[0])
    print(dataset.to_cluster_names([1, 5, 10]))
    print(dataset.code_to_wo)
