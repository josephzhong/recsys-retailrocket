from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

from tqdm import tqdm

def download_kaggle_dataset(
    dataset_name: str,
    output_dir: str | Path = "data",
    force: bool = False,
) -> Path:
    """Download a Kaggle dataset to a local directory with kagglehub."""

    import kagglehub

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download(
        dataset_name,
        force_download=force,
        output_dir=str(destination),
    )
    return Path(path).resolve()


def resolve_category_tree_path(
    dataset_path: str | Path,
    filename: str = "category_tree.csv",
) -> Path:
    """Resolve the category tree CSV path from either a dataset directory or file path."""

    path = Path(dataset_path).expanduser().resolve()
    if path.is_file():
        return path

    direct_match = path / filename
    if direct_match.exists():
        return direct_match

    matches = sorted(path.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} under {path}")
    return matches[0]


def resolve_dataset_file_path(
    dataset_path: str | Path,
    filename: str,
) -> Path:
    """Resolve a dataset file from either a dataset directory or direct file path."""

    return resolve_category_tree_path(dataset_path, filename=filename)


def _parse_single_numeric_property_value(raw_value: str) -> float | None:
    """Parse a property value only when the whole field is one numeric token."""

    tokens = raw_value.split()
    if len(tokens) != 1:
        return None

    token = tokens[0]
    if not token.startswith("n"):
        return None

    try:
        return float(token[1:])
    except ValueError:
        return None


def _split_property_value_tokens(raw_value: str) -> tuple[str, str]:
    """Split property value tokens into numeric-prefixed and non-numeric groups."""

    num_values: list[str] = []
    tokens: list[str] = []

    for token in raw_value.split():
        if token.startswith("n"):
            num_values.append(token)
        else:
            tokens.append(token)

    return " ".join(num_values), " ".join(tokens)


def _is_category_property(property_id: int, category_property_ids: set[int]) -> bool:
    """Return whether a mapped property id should be treated as category-like."""

    return property_id in category_property_ids


def _merge_property_rows(
    rows: list[dict[str, int | str]],
    output_path: Path,
    value_columns: list[str],
) -> int:
    """Merge rows by itemid and property, concatenating timestamp-value pairs."""

    merged_rows: dict[tuple[int, int], dict[str, list[str] | int]] = {}

    for row in rows:
        key = (row["itemid"], row["property"])
        merged_row = merged_rows.setdefault(
            key,
            {
                "itemid": row["itemid"],
                "property": row["property"],
                **{column: [] for column in value_columns},
            },
        )

        for column in value_columns:
            value = row[column]
            merged_row[column].append(f'{row["timestamp"]}|{value}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=["itemid", "property", *value_columns])
        writer.writeheader()

        def sort_key(item: tuple[tuple[int, int], dict[str, list[str] | int]]) -> tuple[int, int]:
            item_id, property_id = item[0]
            return (item_id, property_id)

        for _, merged_row in sorted(merged_rows.items(), key=sort_key):
            writer.writerow(
                {
                    "itemid": merged_row["itemid"],
                    "property": merged_row["property"],
                    **{
                        column: "+".join(merged_row[column])
                        for column in value_columns
                    },
                }
            )

    return len(merged_rows)


def _sort_and_deduplicate_group_rows(
    rows_by_item_property: dict[tuple[int, int], list[dict[str, int | str]]],
    drop_value: bool = False,
) -> list[dict[str, int | str]]:
    """Sort rows within each item-property group and drop consecutive duplicate values."""

    cleaned_rows: list[dict[str, int | str]] = []

    for (item_id, property_id), grouped_rows in tqdm(
        rows_by_item_property.items(),
        desc="Deduplicating item-property groups",
        unit="group",
    ):
        sorted_rows = sorted(grouped_rows, key=lambda row: int(row["timestamp"]))
        previous_value: str | None = None

        for row in sorted_rows:
            current_value = str(row["value"])
            if previous_value == current_value:
                continue

            cleaned_rows.append(
                {
                    "timestamp": int(row["timestamp"]),
                    "itemid": item_id,
                    "property": property_id,
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"timestamp", *(["value"] if drop_value else [])}
                    },
                }
            )
            previous_value = current_value

    return cleaned_rows


def _compute_mean_std(values: list[float]) -> tuple[float, float]:
    """Compute the sample mean and sample standard deviation."""

    if not values:
        return 0.0, 0.0

    count = len(values)
    mean = sum(values) / count
    if count == 1:
        return mean, 0.0

    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, math.sqrt(variance)


def _property_sort_key(property_id: int | str) -> tuple[int | float, str]:
    """Sort property ids numerically when possible."""

    return (property_id, str(property_id)) if isinstance(property_id, int) else (math.inf, property_id)


def load_property_id_map(property_id_map_path: str | Path) -> dict[str, int]:
    """Load the original-to-mapped property id mapping."""

    path = Path(property_id_map_path).expanduser().resolve()
    mapping: dict[str, int] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["original_property_id"]] = int(row["mapped_property_id"])

    return mapping


def load_category_tree(category_tree_path: str | Path) -> dict[int, int | None]:
    """Load the category tree as a mapping of category id to parent id."""

    path = Path(category_tree_path).expanduser().resolve()
    category_tree: dict[int, int | None] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category_id = int(row["categoryid"])
            parent_value = (row.get("parentid") or "").strip()
            category_tree[category_id] = int(parent_value) if parent_value else None

    return category_tree


def build_category_lookup_rows(
    category_tree: dict[int, int | None],
) -> tuple[list[str], list[list[str]]]:
    """Build flattened category lineage rows from a parent-child category tree."""

    lineage_cache: dict[int, tuple[int, ...]] = {}

    def build_lineage(category_id: int, visiting: set[int] | None = None) -> tuple[int, ...]:
        if category_id in lineage_cache:
            return lineage_cache[category_id]

        if visiting is None:
            visiting = set()
        if category_id in visiting:
            raise ValueError(f"Cycle detected in category tree at category {category_id}")

        parent_id = category_tree.get(category_id)
        current_visiting = visiting | {category_id}

        if parent_id is None:
            lineage = (category_id,)
        elif parent_id in category_tree:
            lineage = (*build_lineage(parent_id, current_visiting), category_id)
        else:
            lineage = (parent_id, category_id)

        lineage_cache[category_id] = lineage
        return lineage

    lineages = {category_id: build_lineage(category_id) for category_id in category_tree}
    max_depth = max(len(lineage) for lineage in lineages.values())
    header = [f"level_{level}_category_id" for level in range(1, max_depth + 1)]

    def sort_key(lineage: tuple[int, ...]) -> tuple[int, ...]:
        padded_lineage = list(lineage)
        padded_lineage.extend(float("inf") for _ in range(max_depth - len(lineage)))
        return tuple(padded_lineage)

    rows: list[list[str]] = []
    for _, lineage in sorted(lineages.items(), key=lambda item: sort_key(item[1])):
        row = [str(value) for value in lineage]
        row.extend("" for _ in range(max_depth - len(lineage)))
        rows.append(row)

    return header, rows


def compute_leaf_depth_statistics(category_tree: dict[int, int | None]) -> dict[int, int]:
    """Count leaf categories by their depth from the root."""

    children_by_parent: dict[int, set[int]] = {}
    for category_id, parent_id in category_tree.items():
        if parent_id is not None and parent_id in category_tree:
            children_by_parent.setdefault(parent_id, set()).add(category_id)

    lineage_cache: dict[int, int] = {}

    def get_depth(category_id: int, visiting: set[int] | None = None) -> int:
        if category_id in lineage_cache:
            return lineage_cache[category_id]

        if visiting is None:
            visiting = set()
        if category_id in visiting:
            raise ValueError(f"Cycle detected in category tree at category {category_id}")

        parent_id = category_tree.get(category_id)
        current_visiting = visiting | {category_id}

        if parent_id is None or parent_id not in category_tree:
            depth = 1
        else:
            depth = get_depth(parent_id, current_visiting) + 1

        lineage_cache[category_id] = depth
        return depth

    leaf_category_ids = [
        category_id for category_id in category_tree if category_id not in children_by_parent
    ]
    depth_counts = Counter(get_depth(category_id) for category_id in leaf_category_ids)
    return dict(sorted(depth_counts.items()))


def generate_category_lookup_table(
    dataset_path: str | Path,
    output_path: str | Path | None = None,
    category_tree_filename: str = "category_tree.csv",
) -> tuple[Path, dict[int, int]]:
    """Generate a flattened category lookup CSV and leaf-depth stats from the category tree."""

    category_tree_path = resolve_category_tree_path(
        dataset_path,
        filename=category_tree_filename,
    )
    category_tree = load_category_tree(category_tree_path)
    header, rows = build_category_lookup_rows(category_tree)
    leaf_depth_statistics = compute_leaf_depth_statistics(category_tree)

    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else category_tree_path.with_name("category_lookup.csv")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    return destination, leaf_depth_statistics


def merge_item_properties_files(
    dataset_path: str | Path,
    output_path: str | Path | None = None,
    part1_filename: str = "item_properties_part1.csv",
    part2_filename: str = "item_properties_part2.csv",
) -> Path:
    """Merge the Retailrocket item properties parts into a single CSV."""

    part1_path = resolve_dataset_file_path(dataset_path, filename=part1_filename)
    part2_path = resolve_dataset_file_path(dataset_path, filename=part2_filename)

    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else part1_path.with_name("item_properties.csv")
    )
    property_id_map_path = destination.with_name("property_id_map.csv")

    if destination.exists() and property_id_map_path.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "itemid", "property", "value"]
    merged_rows: list[dict[str, str]] = []

    for source_path in (part1_path, part2_path):
        with source_path.open(newline="", encoding="utf-8") as input_handle:
            reader = csv.DictReader(input_handle)

            if reader.fieldnames is None:
                continue

            if reader.fieldnames != fieldnames:
                raise ValueError(
                    f"CSV header mismatch for {source_path.name}: "
                    f"expected {fieldnames}, got {reader.fieldnames}"
                )

            merged_rows.extend(reader)

    unique_property_ids = list({row["property"] for row in merged_rows})
    property_id_mapping = {
        property_id: mapped_property_id
        for mapped_property_id, property_id in enumerate(unique_property_ids)
    }

    with property_id_map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_property_id", "mapped_property_id"])
        for property_id in unique_property_ids:
            writer.writerow([property_id, property_id_mapping[property_id]])

    integer_rows: list[dict[str, int | str]] = []
    for row in merged_rows:
        integer_rows.append(
            {
                "timestamp": int(row["timestamp"]),
                "itemid": int(row["itemid"]),
                "property": property_id_mapping[row["property"]],
                "value": row["value"],
            }
        )

    with destination.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(integer_rows)

    return destination


def load_item_property_row_groups(
    item_properties_path: str | Path,
) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]], list[dict[str, int | str]]]:
    """Read item properties once and split rows into numeric, category, and non-numeric groups."""

    path = Path(item_properties_path).expanduser().resolve()
    property_id_mapping = load_property_id_map(path.with_name("property_id_map.csv"))
    category_property_ids = {
        property_id_mapping["categoryid"],
        property_id_mapping["available"],
    }

    numeric_rows: list[dict[str, int | str]] = []
    category_rows: list[dict[str, int | str]] = []
    non_numeric_rows: list[dict[str, int | str]] = []
    grouped_numeric_rows: dict[tuple[int, int], list[dict[str, int | str]]] = {}
    grouped_category_rows: dict[tuple[int, int], list[dict[str, int | str]]] = {}
    grouped_non_numeric_rows: dict[tuple[int, int], list[dict[str, int | str]]] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(reader, desc="Loading item property rows", unit="row"):
            timestamp = int(row["timestamp"])
            item_id = int(row["itemid"])
            property_id = int(row["property"])
            raw_value = str(row["value"])
            item_property_key = (item_id, property_id)

            if _is_category_property(property_id, category_property_ids):
                grouped_category_rows.setdefault(item_property_key, []).append(
                    {
                        "timestamp": timestamp,
                        "value": raw_value,
                    }
                )
                continue

            if _parse_single_numeric_property_value(raw_value) is not None:
                grouped_numeric_rows.setdefault(item_property_key, []).append(
                    {
                        "timestamp": timestamp,
                        "value": raw_value,
                    }
                )
                continue

            num_values, tokens = _split_property_value_tokens(raw_value)
            grouped_non_numeric_rows.setdefault(item_property_key, []).append(
                {
                    "timestamp": timestamp,
                    "value": raw_value,
                    "num_values": num_values,
                    "tokens": tokens,
                }
            )

    numeric_rows = _sort_and_deduplicate_group_rows(grouped_numeric_rows)
    category_rows = _sort_and_deduplicate_group_rows(grouped_category_rows)
    non_numeric_rows = _sort_and_deduplicate_group_rows(
        grouped_non_numeric_rows,
        drop_value=True,
    )

    return numeric_rows, category_rows, non_numeric_rows


def process_numeric_property_values(
    item_properties_path: str | Path,
    numeric_rows: list[dict[str, int | str]],
    numeric_properties_filename: str = "numeric_properties.csv",
) -> dict[str, Path | int]:
    """Write normalized merged numeric property rows and per-property stats."""

    item_properties_path = Path(item_properties_path).expanduser().resolve()
    numeric_properties_path = item_properties_path.with_name(numeric_properties_filename)
    property_mean_std_path = item_properties_path.with_name("property_mean_std.csv")

    def get_statistics_numeric_property_values(
        rows: list[dict[str, int | str]],
    ) -> dict[int, dict[str, float | int]]:
        """Build property-level numeric statistics and write them to CSV."""

        values_by_item_property: dict[tuple[int, int], list[float]] = {}
        for row in rows:
            numeric_value = _parse_single_numeric_property_value(row["value"])
            if numeric_value is None:
                continue

            key = (int(row["itemid"]), int(row["property"]))
            values_by_item_property.setdefault(key, []).append(numeric_value)

        property_statistics: dict[int, dict[str, float | int | list[float]]] = {}

        for (_, property_id), values in values_by_item_property.items():
            original_values = list(values)
            added_values: list[float] = []

            if len(original_values) == 1:
                added_values = [original_values[0]] * 3

            summary = property_statistics.setdefault(
                property_id,
                {
                    "original_value_count": 0,
                    "added_value_count": 0,
                    "values": [],
                },
            )
            summary["original_value_count"] += len(original_values)
            summary["added_value_count"] += len(added_values)
            summary["values"].extend(original_values)
            summary["values"].extend(added_values)

        property_mean_std_path.parent.mkdir(parents=True, exist_ok=True)
        with property_mean_std_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "propertyid",
                    "original_value_count",
                    "added_value_count",
                    "total_value_count",
                    "mean_value",
                    "std_value",
                ]
            )

            for property_id in sorted(property_statistics, key=_property_sort_key):
                summary = property_statistics[property_id]
                values = summary["values"]
                mean_value, std_value = _compute_mean_std(values)
                writer.writerow(
                    [
                        property_id,
                        summary["original_value_count"],
                        summary["added_value_count"],
                        len(values),
                        mean_value,
                        std_value,
                    ]
                )
                summary["mean_value"] = mean_value
                summary["std_value"] = std_value

        return {
            property_id: {
                "original_value_count": int(summary["original_value_count"]),
                "added_value_count": int(summary["added_value_count"]),
                "mean_value": float(summary["mean_value"]),
                "std_value": float(summary["std_value"]),
            }
            for property_id, summary in property_statistics.items()
        }

    def verify_property_mean_std_counts(rows: list[dict[str, int | str]]) -> dict[str, int]:
        """Compare numeric row counts against property_mean_std.csv."""

        item_property_line_counts: dict[int, int] = {}
        for row in rows:
            property_id = int(row["property"])
            item_property_line_counts[property_id] = item_property_line_counts.get(property_id, 0) + 1

        property_mean_std_counts: dict[int, int] = {}
        with property_mean_std_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                property_mean_std_counts[int(row["propertyid"])] = int(row["original_value_count"])

        property_ids = sorted(
            set(item_property_line_counts) | set(property_mean_std_counts),
            key=_property_sort_key,
        )

        matching_property_count = 0
        mismatching_property_count = 0

        for property_id in property_ids:
            source_count = item_property_line_counts.get(property_id, 0)
            stats_count = property_mean_std_counts.get(property_id, 0)
            if source_count == stats_count:
                matching_property_count += 1
            else:
                mismatching_property_count += 1

        return {
            "property_count": len(property_ids),
            "matching_property_count": matching_property_count,
            "mismatching_property_count": mismatching_property_count,
        }

    property_statistics = get_statistics_numeric_property_values(numeric_rows)
    verification_summary = verify_property_mean_std_counts(numeric_rows)

    normalized_numeric_rows: list[dict[str, int | str]] = []
    for row in numeric_rows:
        property_id = int(row["property"])
        numeric_value = _parse_single_numeric_property_value(str(row["value"]))
        if numeric_value is None:
            continue

        mean_value = float(property_statistics[property_id]["mean_value"])
        std_value = float(property_statistics[property_id]["std_value"])
        normalized_value = 0.0 if std_value == 0.0 else (numeric_value - mean_value) / std_value

        normalized_numeric_rows.append(
            {
                **row,
                "value": f"{normalized_value:.12g}",
            }
        )

    numeric_row_count = _merge_property_rows(
        rows=normalized_numeric_rows,
        output_path=numeric_properties_path,
        value_columns=["value"],
    )

    return {
        "numeric_properties_path": numeric_properties_path,
        "property_mean_std_path": property_mean_std_path,
        "numeric_property_row_count": numeric_row_count,
        "property_count": verification_summary["property_count"],
        "matching_property_count": verification_summary["matching_property_count"],
        "mismatching_property_count": verification_summary["mismatching_property_count"],
    }


def process_non_numeric_property_values(
    item_properties_path: str | Path,
    non_numeric_rows: list[dict[str, int | str]],
    output_filename: str = "non_numeric_properties.csv",
) -> dict[str, Path | int]:
    """Write merged non-numeric property rows with timestamp-value pairs."""

    item_properties_path = Path(item_properties_path).expanduser().resolve()
    output_path = item_properties_path.with_name(output_filename)
    non_numeric_row_count = _merge_property_rows(
        rows=non_numeric_rows,
        output_path=output_path,
        value_columns=["value", "num_values", "tokens"],
    )

    return {
        "non_numeric_properties_path": output_path,
        "non_numeric_property_row_count": non_numeric_row_count,
    }


def process_category_property_values(
    item_properties_path: str | Path,
    category_rows: list[dict[str, int | str]],
    output_filename: str = "cate_properties.csv",
) -> dict[str, Path | int]:
    """Write merged category-like property rows with timestamp-value pairs."""

    item_properties_path = Path(item_properties_path).expanduser().resolve()
    output_path = item_properties_path.with_name(output_filename)
    category_property_row_count = _merge_property_rows(
        rows=category_rows,
        output_path=output_path,
        value_columns=["value"],
    )

    return {
        "cate_properties_path": output_path,
        "category_property_row_count": category_property_row_count,
    }


def preprocess_retailrocket_data(dataset_path: str | Path) -> dict[str, Path | dict[int, int]]:
    """Run the Retailrocket preprocessing steps and return the generated outputs."""

    category_lookup_path, leaf_depth_statistics = generate_category_lookup_table(dataset_path)
    print(f"Finished generate_category_lookup_table -> {category_lookup_path}")

    merged_item_properties_path = merge_item_properties_files(dataset_path)
    print(f"Finished merge_item_properties_files -> {merged_item_properties_path}")

    numeric_rows, category_rows, non_numeric_rows = load_item_property_row_groups(
        merged_item_properties_path
    )
    print(
        "Finished load_item_property_row_groups -> "
        f"numeric={len(numeric_rows):,}, "
        f"category={len(category_rows):,}, "
        f"non_numeric={len(non_numeric_rows):,}"
    )

    numeric_property_outputs = process_numeric_property_values(
        merged_item_properties_path,
        numeric_rows,
    )
    print(
        "Finished process_numeric_property_values -> "
        f'{numeric_property_outputs["numeric_properties_path"]}'
    )

    category_property_outputs = process_category_property_values(
        merged_item_properties_path,
        category_rows,
    )
    print(
        "Finished process_category_property_values -> "
        f'{category_property_outputs["cate_properties_path"]}'
    )

    non_numeric_property_outputs = process_non_numeric_property_values(
        merged_item_properties_path,
        non_numeric_rows,
    )
    print(
        "Finished process_non_numeric_property_values -> "
        f'{non_numeric_property_outputs["non_numeric_properties_path"]}'
    )

    return {
        "category_lookup_path": category_lookup_path,
        "leaf_depth_statistics": leaf_depth_statistics,
        "merged_item_properties_path": merged_item_properties_path,
        "property_id_map_path": merged_item_properties_path.with_name("property_id_map.csv"),
        "numeric_properties_path": numeric_property_outputs["numeric_properties_path"],
        "property_mean_std_path": numeric_property_outputs["property_mean_std_path"],
        "cate_properties_path": category_property_outputs["cate_properties_path"],
        "non_numeric_properties_path": non_numeric_property_outputs["non_numeric_properties_path"],
    }


__all__ = [
    "build_category_lookup_rows",
    "compute_leaf_depth_statistics",
    "download_kaggle_dataset",
    "generate_category_lookup_table",
    "load_item_property_row_groups",
    "load_category_tree",
    "load_property_id_map",
    "process_category_property_values",
    "merge_item_properties_files",
    "process_non_numeric_property_values",
    "process_numeric_property_values",
    "preprocess_retailrocket_data",
    "resolve_dataset_file_path",
    "resolve_category_tree_path",
]
