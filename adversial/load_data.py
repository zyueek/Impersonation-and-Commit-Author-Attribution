from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


LabelField = Literal["emailname", "author", "label"]
AuthorField = Literal["emailname", "author", "none"]


@dataclass(frozen=True)
class LoadedDataset:
    df: pd.DataFrame
    label_field: LabelField
    author_field: AuthorField


def _join_code_lines(item: dict) -> str:
    added = item.get("added_code", "")
    if isinstance(added, list):
        return "\n".join(map(str, added))
    return str(added or "")


def detect_label_field(sample_item: dict[str, Any], *, task: str) -> LabelField:
    if task == "binary":
        if "label" not in sample_item:
            raise ValueError("task=binary requires 'label' field in JSON items")
        return "label"
    if "emailname" in sample_item and sample_item.get("emailname"):
        return "emailname"
    if "author" in sample_item and sample_item.get("author"):
        return "author"
    if "emailname" in sample_item:
        return "emailname"
    if "author" in sample_item:
        return "author"
    raise ValueError("Could not detect label field (expected 'emailname' or 'author')")


def detect_author_field(sample_item: dict[str, Any]) -> AuthorField:
    if "emailname" in sample_item:
        return "emailname"
    if "author" in sample_item:
        return "author"
    return "none"


def load_commit_json(
    json_path: str,
    *,
    task: str,
    min_samples_per_author: int = 2,
) -> LoadedDataset:
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or (data and not isinstance(data[0], dict)):
        raise ValueError(f"Unexpected JSON format: {json_path}")
    if not data:
        raise ValueError(f"Empty JSON: {json_path}")

    label_field = detect_label_field(data[0], task=task)
    author_field = detect_author_field(data[0])

    records = []
    for item in data:
        code = _join_code_lines(item)
        author_id = None
        if author_field != "none":
            author_id = item.get(author_field)
        records.append(
            {
                "code": code,
                "message": str(item.get("message", "") or ""),
                "filename": str(item.get("filename", "") or ""),
                "label": item.get(label_field),
                "author_id": author_id,
                "raw": item,
            }
        )
    df = pd.DataFrame(records)
    df = df[df["code"].astype(bool)]

    # Label filtering:
    # - authorship: drop empty ids
    # - binary: keep 0/False labels, only drop missing
    if task == "authorship":
        df = df[df["label"].astype(str).str.len() > 0]
    else:
        df = df[df["label"].notna()]

    if task == "authorship":
        counts = df["label"].value_counts()
        keep = counts[counts >= min_samples_per_author].index
        df = df[df["label"].isin(keep)].copy()

    df = df.reset_index(drop=True)
    return LoadedDataset(df=df, label_field=label_field, author_field=author_field)
