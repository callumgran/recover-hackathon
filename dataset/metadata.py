import os
import sys
import typing
from pathlib import Path
from urllib.error import URLError

import polars as pl
import torch
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.base import BaseDataset, index_encode_str


class MetadataDataset(BaseDataset):
    competition = "hackathon-recover-x-cogito"
    resources: typing.ClassVar[list[str]] = [
        "metaData.csv",
    ]

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        download: bool = False,
        num_companies: int = 14,  # Number of insurance companies to encode
    ):
        super().__init__(root)
        self.split = split
        self.num_companies = num_companies

        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, pid):
        idx = self.data.get_column("project_id").index_of(pid)  # type: ignore
        if idx is None:
            raise KeyError(f"Project ID {pid} not found in dataset.")
        row = self.data[idx]
        return {
            "project_id": row["project_id"][0],
            "insurance_company": row["insurance_company"][0],
            "insurance_company_one_hot": torch.tensor(row["insurance_company_one_hot"], dtype=torch.int8),
            "recover_office_zip_code": int(row["recover_office_zip_code"][0] or 0),
            "damage_address_zip_code": int(row["damage_address_zip_code"][0] or 0),
            "office_distance": float(row["office_distance"][0] or 0),
            "case_creation_year": int(row["case_creation_year"][0] or 0),
            "case_creation_month": int(row["case_creation_month"][0] or 0),
        }

    def _check_exists(self) -> bool:
        return all(os.path.exists(os.path.join(self.data_folder, filename)) for filename in self.resources)

    @property
    def data_folder(self) -> str:
        return self.root

    def _load_data(self):
        lf = (
            pl.scan_csv(
                os.path.join(self.data_folder, "metaData.csv"),
                schema={
                    "insurance_company": pl.Utf8,
                    "recover_office_zip_code": pl.Utf8,
                    "damage_address_zip_code": pl.Utf8,
                    "office_distance": pl.Utf8,
                    "case_creation_year": pl.Int32,
                    "case_creation_month": pl.Utf8,
                    "project_id": pl.Int64,
                },
            )
            .unique("project_id")
            .with_columns(
                pl.col("office_distance").str.replace_all(",", ".").cast(pl.Float32).alias("office_distance"),
            )
        )

        df = lf.collect()

        company_lookup = {chr(k): k - ord("A") for k in range(ord("A"), ord("A") + self.num_companies)}
        df = df.with_columns(
            pl.col("insurance_company")
            .map_elements(
                lambda x: index_encode_str(x, len(company_lookup), company_lookup),
                return_dtype=pl.List(pl.Int8),
            )
            .alias("insurance_company_one_hot")
        )

        self.data = df


if __name__ == "__main__":
    dataset = MetadataDataset(root="data", split="train", download=True)
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[dataset.data.get_column("project_id")[0]]
    print(f"Sample data: {sample}")
