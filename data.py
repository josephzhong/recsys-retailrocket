from __future__ import annotations

import csv
import math
import random
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from tqdm import tqdm
from model import get_item_embedding, get_user_embedding


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


def _add_months(value: date, months: int) -> date:
    """Add calendar months to a date, clamping to the last valid day when needed."""

    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1

    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)

    last_day = (next_month - timedelta(days=1)).day
    return date(year, month, min(value.day, last_day))


def generate_events_time_range(
    dataset_path: str | Path,
    output_filename: str = "events_time_range.csv",
) -> dict[str, Path | int]:
    """Read events.csv, compute global and per-item timestamp ranges, and write them to a CSV file."""

    events_path = resolve_dataset_file_path(dataset_path, "events.csv")
    output_path = events_path.with_name(output_filename)

    min_timestamp: int | None = None
    max_timestamp: int | None = None
    item_time_ranges: dict[int, dict[str, int | None]] = {}

    with events_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(reader, desc="Scanning event timestamps", unit="row"):
            timestamp = int(row["timestamp"])
            item_id = int(row["itemid"])
            event_name = row["event"]
            if min_timestamp is None or timestamp < min_timestamp:
                min_timestamp = timestamp
            if max_timestamp is None or timestamp > max_timestamp:
                max_timestamp = timestamp
            item_time_range = item_time_ranges.setdefault(
                item_id,
                {
                    "min_timestamp": None,
                    "max_timestamp": timestamp,
                },
            )
            if timestamp > item_time_range["max_timestamp"]:
                item_time_range["max_timestamp"] = timestamp
            if event_name == "view" and (
                item_time_range["min_timestamp"] is None
                or timestamp < item_time_range["min_timestamp"]
            ):
                item_time_range["min_timestamp"] = timestamp

    if min_timestamp is None or max_timestamp is None:
        raise ValueError(f"No event rows found in {events_path}")

    min_date_value = datetime.fromtimestamp(min_timestamp / 1000, tz=UTC).date()
    max_date_value = datetime.fromtimestamp(max_timestamp / 1000, tz=UTC).date()
    cutoff_date = _add_months(min_date_value, 4).strftime("%Y-%m-%d")
    min_date = min_date_value.strftime("%Y-%m-%d")
    max_date = max_date_value.strftime("%Y-%m-%d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "itemid",
                "min_timestamp",
                "max_timestamp",
                "min_date",
                "max_date",
                "cutoff_date",
                "is_cold_start_item",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scope": "global",
                "itemid": "",
                "min_timestamp": min_timestamp,
                "max_timestamp": max_timestamp,
                "min_date": min_date,
                "max_date": max_date,
                "cutoff_date": cutoff_date,
                "is_cold_start_item": "",
            }
        )
        for item_id, item_time_range in sorted(item_time_ranges.items()):
            item_min_timestamp = item_time_range["min_timestamp"]
            item_min_date = (
                datetime.fromtimestamp(item_min_timestamp / 1000, tz=UTC).strftime("%Y-%m-%d")
                if item_min_timestamp is not None
                else ""
            )
            item_max_date = datetime.fromtimestamp(
                item_time_range["max_timestamp"] / 1000,
                tz=UTC,
            ).strftime("%Y-%m-%d")
            is_cold_start_item = int(bool(item_min_date) and item_min_date >= cutoff_date)
            writer.writerow(
                {
                    "scope": "item",
                    "itemid": item_id,
                    "min_timestamp": item_min_timestamp if item_min_timestamp is not None else "",
                    "max_timestamp": item_time_range["max_timestamp"],
                    "min_date": item_min_date,
                    "max_date": item_max_date,
                    "cutoff_date": cutoff_date,
                    "is_cold_start_item": is_cold_start_item,
                }
            )

    return {
        "events_time_range_path": output_path,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "min_date": min_date,
        "max_date": max_date,
        "cutoff_date": cutoff_date,
    }


def _cutoff_date_to_timestamp_ms(cutoff_date: str) -> int:
    """Convert a YYYY-MM-DD cutoff date to a UTC millisecond timestamp."""

    cutoff_datetime = datetime.strptime(cutoff_date, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(cutoff_datetime.timestamp() * 1000)


def _load_item_cold_start_labels(events_time_range_path: str | Path) -> dict[int, int]:
    """Load per-item cold-start labels from events_time_range.csv."""

    path = Path(events_time_range_path).expanduser().resolve()
    item_cold_start_labels: dict[int, int] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["scope"] != "item":
                continue
            item_cold_start_labels[int(row["itemid"])] = int(row["is_cold_start_item"])

    return item_cold_start_labels


def _load_item_available_values(
    cate_properties_path: str | Path,
    property_id_map_path: str | Path,
    cutoff_timestamp: int,
) -> dict[int, str]:
    """Load item availability at the cutoff from cate_properties.csv."""

    property_id_map_path = Path(property_id_map_path).expanduser().resolve()
    available_property_id: str | None = None

    with property_id_map_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["original_property_id"] == "available" and row.get("property_type") == "0":
                available_property_id = row["mapped_property_id"]
                break

    if available_property_id is None:
        raise ValueError(f"Could not find mapped category property id for 'available' in {property_id_map_path}")

    cate_properties_path = Path(cate_properties_path).expanduser().resolve()
    item_available_values: dict[int, str] = {}

    with cate_properties_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["property"] != available_property_id:
                continue

            history = []
            for timestamp_value in str(row["value"]).split("+"):
                timestamp_str, value = timestamp_value.split("|", maxsplit=1)
                history.append((int(timestamp_str), value))

            last_available_one_timestamp: int | None = None
            max_timestamp = history[-1][0]
            for timestamp, value in history:
                if value == "1":
                    last_available_one_timestamp = timestamp

            if last_available_one_timestamp is None:
                item_available_values[int(row["itemid"])] = "0"
                continue

            if last_available_one_timestamp == max_timestamp:
                item_available_values[int(row["itemid"])] = "1"
                continue

            next_zero_timestamp: int | None = None
            for timestamp, value in history:
                if timestamp > last_available_one_timestamp and value == "0":
                    next_zero_timestamp = timestamp
                    break

            if next_zero_timestamp is None or next_zero_timestamp > cutoff_timestamp:
                item_available_values[int(row["itemid"])] = "1"
            else:
                item_available_values[int(row["itemid"])] = "0"

    return item_available_values


def _event_timestamp_to_bucket_idx(timestamp_ms: int, cutoff_timestamp_ms: int) -> int:
    """Map a pre-cutoff event timestamp to a cutoff-relative day bucket."""

    day_in_ms = 24 * 60 * 60 * 1000
    elapsed_ms = max(0, cutoff_timestamp_ms - timestamp_ms)

    if elapsed_ms < day_in_ms:
        return 5
    if elapsed_ms < 3 * day_in_ms:
        return 4
    if elapsed_ms < 7 * day_in_ms:
        return 3
    if elapsed_ms < 15 * day_in_ms:
        return 2
    if elapsed_ms < 30 * day_in_ms:
        return 1
    return 0


def merge_events_by_visitor_item(
    dataset_path: str | Path,
    output_filename: str = "user_item.csv",
    user_output_filename: str = "user_events.csv",
    cutoff_date: str | None = None,
) -> dict[str, Path | int | str]:
    """Write user-item labels and visitor-level event summaries."""

    events_path = resolve_dataset_file_path(dataset_path, "events.csv")
    output_path = events_path.with_name(output_filename)
    user_output_path = events_path.with_name(user_output_filename)
    dataset_dir = events_path.parent
    if cutoff_date is None:
        time_range = generate_events_time_range(dataset_path)
        cutoff_date = str(time_range["cutoff_date"])
    else:
        time_range = generate_events_time_range(dataset_path)
    item_cold_start_labels = _load_item_cold_start_labels(time_range["events_time_range_path"])
    cutoff_timestamp = _cutoff_date_to_timestamp_ms(cutoff_date)
    item_available_values = _load_item_available_values(
        dataset_dir / "cate_properties.csv",
        dataset_dir / "property_id_map.csv",
        cutoff_timestamp,
    )
    event_type_to_id = {
        "view": 0,
        "addtocart": 1,
        "transaction": 2,
    }
    user_action_column_names = {
        0: "by_user_view_events_lists",
        1: "by_user_cart_events_lists",
        2: "by_user_transaction_events_lists",
    }

    grouped_events: dict[tuple[int, int], list[tuple[int, int]]] = {}

    with events_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(reader, desc="Grouping events by visitor-item", unit="row"):
            event_name = row["event"]
            if event_name not in event_type_to_id:
                raise ValueError(f"Unsupported event type {event_name!r} in {events_path}")

            key = (int(row["visitorid"]), int(row["itemid"]))
            grouped_events.setdefault(key, []).append(
                (int(row["timestamp"]), event_type_to_id[event_name])
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_row_count = 0
    negative_data_count = 0
    positive_data_count = 0
    positive_cold_start_data_count = 0
    negative_cold_start_data_count = 0
    merged_rows: list[dict[str, int | str]] = []
    visitor_event_counts: dict[int, dict[int, Counter[tuple[int, int]]]] = {}

    for (visitor_id, item_id), events in tqdm(
        sorted(grouped_events.items()),
        desc="Building merged visitor-item events",
        unit="group",
    ):
        sorted_events = sorted(events)
        label = int(
            any(
                event_id == 0 and timestamp >= cutoff_timestamp
                for timestamp, event_id in sorted_events
            )
        )
        item_bucket_counts_by_action = {
            0: Counter(),
            1: Counter(),
            2: Counter(),
        }
        for timestamp, event_id in sorted_events:
            if timestamp >= cutoff_timestamp:
                continue
            bucket_idx = _event_timestamp_to_bucket_idx(timestamp, cutoff_timestamp)
            item_bucket_counts_by_action[event_id][bucket_idx] += 1

        is_cold_start_item = item_cold_start_labels.get(item_id, 0)
        available = item_available_values.get(item_id, "")

        merged_rows.append(
            {
                "visitorid": visitor_id,
                "itemid": item_id,
                "available": available,
                "label": label,
                "is_cold_start_item": is_cold_start_item,
            }
        )
        visitor_action_counters = visitor_event_counts.setdefault(
            visitor_id,
            {
                0: Counter(),
                1: Counter(),
                2: Counter(),
            },
        )
        for event_id, item_bucket_counts in item_bucket_counts_by_action.items():
            for bucket_idx, count in item_bucket_counts.items():
                visitor_action_counters[event_id][(item_id, bucket_idx)] += count
        merged_row_count += 1
        if label == 0:
            negative_data_count += 1
            if is_cold_start_item == 1:
                negative_cold_start_data_count += 1
        else:
            positive_data_count += 1
            if is_cold_start_item == 1:
                positive_cold_start_data_count += 1

    by_user_events_lists = {
        visitor_id: {
            event_id: "+".join(
                f"{item_id}|{bucket_idx}|{count}"
                for (item_id, bucket_idx), count in sorted(item_event_counts[event_id].items())
            )
            for event_id in user_action_column_names
        }
        for visitor_id, item_event_counts in tqdm(
            visitor_event_counts.items(),
            desc="Building visitor-level event lists",
            unit="visitor",
        )
    }

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "visitorid",
                "itemid",
                "available",
                "label",
                "is_cold_start_item",
            ],
        )
        writer.writeheader()

        for row in tqdm(merged_rows, desc="Writing merged visitor-item events", unit="row"):
            writer.writerow(
                {
                    "visitorid": row["visitorid"],
                    "itemid": row["itemid"],
                    "available": row["available"],
                    "label": row["label"],
                    "is_cold_start_item": row["is_cold_start_item"],
                }
            )

    with user_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "visitorid",
                "by_user_view_events_lists",
                "by_user_cart_events_lists",
                "by_user_transaction_events_lists",
            ],
        )
        writer.writeheader()
        for visitor_id, events_by_action in tqdm(
            sorted(by_user_events_lists.items()),
            desc="Writing user event lists",
            unit="user",
        ):
            writer.writerow(
                {
                    "visitorid": visitor_id,
                    "by_user_view_events_lists": events_by_action[0],
                    "by_user_cart_events_lists": events_by_action[1],
                    "by_user_transaction_events_lists": events_by_action[2],
                }
            )

    return {
        "user_item_path": output_path,
        "user_events_path": user_output_path,
        "user_item_row_count": merged_row_count,
        "negative_data_count": negative_data_count,
        "positive_data_count": positive_data_count,
        "negative_cold_start_data_count": negative_cold_start_data_count,
        "positive_cold_start_data_count": positive_cold_start_data_count,
        "cutoff_date": cutoff_date,
        "events_merged_path": output_path,
        "events_merged_row_count": merged_row_count,
    }


def _parse_single_numeric_property_value(raw_value: str) -> float | None:
    """Parse a property value only when the whole field is one numeric token."""

    tokens = raw_value.split()
    if len(tokens) != 1:
        return None

    token = tokens[0]
    if not token.startswith("n"):
        return None

    try:
        parsed = float(token[1:])
    except ValueError:
        return None

    if not math.isfinite(parsed):
        return None
    return parsed


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


def _property_bucket_upper_bounds(cutoff_timestamp_ms: int) -> dict[int, int]:
    """Return the exclusive upper timestamp bound for each pre-cutoff property bucket."""

    day_in_ms = 24 * 60 * 60 * 1000
    return {
        0: cutoff_timestamp_ms - 30 * day_in_ms,
        1: cutoff_timestamp_ms - 15 * day_in_ms,
        2: cutoff_timestamp_ms - 7 * day_in_ms,
        3: cutoff_timestamp_ms - 3 * day_in_ms,
        4: cutoff_timestamp_ms - day_in_ms,
        5: cutoff_timestamp_ms,
    }


def _build_bucket_value_history(
    history: list[tuple[int, str]],
    cutoff_timestamp_ms: int,
) -> str:
    """Convert timestamp histories into dense bucket histories with a cutoff bucket."""

    upper_bounds = _property_bucket_upper_bounds(cutoff_timestamp_ms)
    sorted_history = sorted(history)

    history_index = 0
    latest_value = "0"
    bucket_pairs: list[str] = []

    for bucket_idx in range(6):
        bucket_upper_bound = upper_bounds[bucket_idx]
        while (
            history_index < len(sorted_history)
            and sorted_history[history_index][0] < bucket_upper_bound
        ):
            latest_value = sorted_history[history_index][1]
            history_index += 1

        bucket_pairs.append(f"{bucket_idx}|{latest_value}")

    while (
        history_index < len(sorted_history)
        and sorted_history[history_index][0] < cutoff_timestamp_ms
    ):
        latest_value = sorted_history[history_index][1]
        history_index += 1

    cutoff_bucket_value = latest_value
    while history_index < len(sorted_history):
        cutoff_bucket_value = sorted_history[history_index][1]
        history_index += 1

    bucket_pairs.append(f"6|{cutoff_bucket_value}")

    return "+".join(bucket_pairs)


def write_bucket_index_property_files(
    dataset_path: str | Path,
    cutoff_date: str,
    numeric_properties_filename: str = "numeric_properties_bucket_idx.csv",
    cate_properties_filename: str = "cate_properties_bucket_idx.csv",
    non_numeric_properties_filename: str = "non_numeric_properties_bucket_idx.csv",
) -> dict[str, Path | int]:
    """Write bucket-index versions of the merged property CSVs without changing the originals."""

    dataset_dir = Path(dataset_path).expanduser().resolve()
    if dataset_dir.is_file():
        dataset_dir = dataset_dir.parent

    cutoff_timestamp = _cutoff_date_to_timestamp_ms(cutoff_date)

    outputs: dict[str, Path | int] = {}
    source_specs = [
        (
            dataset_dir / "numeric_properties.csv",
            dataset_dir / numeric_properties_filename,
            ["value"],
            "numeric_properties_bucket_idx_path",
            "numeric_properties_bucket_idx_row_count",
        ),
        (
            dataset_dir / "cate_properties.csv",
            dataset_dir / cate_properties_filename,
            ["value"],
            "cate_properties_bucket_idx_path",
            "cate_properties_bucket_idx_row_count",
        ),
        (
            dataset_dir / "non_numeric_properties.csv",
            dataset_dir / non_numeric_properties_filename,
            ["num_values", "tokens"],
            "non_numeric_properties_bucket_idx_path",
            "non_numeric_properties_bucket_idx_row_count",
        ),
    ]

    for source_path, output_path, value_columns, path_key, count_key in source_specs:
        row_count = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with source_path.open(newline="", encoding="utf-8") as input_handle, output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_handle:
            reader = csv.DictReader(input_handle)
            writer = csv.DictWriter(output_handle, fieldnames=["itemid", "property", *value_columns])
            writer.writeheader()

            for row in reader:
                output_row = {
                    "itemid": row["itemid"],
                    "property": row["property"],
                }

                for column in value_columns:
                    history: list[tuple[int, str]] = []
                    raw_history = str(row.get(column, ""))

                    if raw_history:
                        for timestamp_value in raw_history.split("+"):
                            timestamp_str, value = timestamp_value.split("|", maxsplit=1)
                            history.append((int(timestamp_str), value))

                    output_row[column] = _build_bucket_value_history(
                        history=history,
                        cutoff_timestamp_ms=cutoff_timestamp,
                    )

                writer.writerow(output_row)
                row_count += 1

        outputs[path_key] = output_path
        outputs[count_key] = row_count

    return outputs


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
            if row.get("property_type") not in (None, "", "0", 0):
                continue
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

    category_property_ids = {"categoryid", "available"}
    property_type_codes = {
        "category": 0,
        "non-numeric": 1,
        "numeric": 2,
    }
    property_type_mapping: dict[str, int] = {}
    property_id_mapping: dict[tuple[str, int], int] = {}
    property_type_counters = {
        0: 0,
        1: 0,
        2: 0,
    }
    property_observations: dict[str, dict[str, bool]] = {}

    for row in merged_rows:
        property_id = row["property"]
        observation = property_observations.setdefault(
            property_id,
            {
                "is_category": property_id in category_property_ids,
                "saw_valid_numeric": False,
                "saw_non_numeric": False,
            },
        )
        if observation["is_category"]:
            continue

        if _parse_single_numeric_property_value(row["value"]) is not None:
            observation["saw_valid_numeric"] = True
        else:
            observation["saw_non_numeric"] = True

    for property_id, observation in property_observations.items():
        if observation["is_category"]:
            property_type = property_type_codes["category"]
        elif observation["saw_valid_numeric"] and not observation["saw_non_numeric"]:
            property_type = property_type_codes["numeric"]
        else:
            property_type = property_type_codes["non-numeric"]

        property_type_mapping[property_id] = property_type
        mapping_key = (property_id, property_type)
        if mapping_key not in property_id_mapping:
            property_id_mapping[mapping_key] = property_type_counters[property_type]
            property_type_counters[property_type] += 1

    with property_id_map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_property_id", "property_type", "mapped_property_id"])
        for property_id in sorted(property_type_mapping, key=_property_sort_key):
            property_type = property_type_mapping[property_id]
            writer.writerow(
                [
                    property_id,
                    property_type,
                    property_id_mapping[(property_id, property_type)],
                ]
            )

    integer_rows: list[dict[str, int | str]] = []
    for row in merged_rows:
        property_id = row["property"]
        property_type = property_type_mapping[property_id]
        integer_rows.append(
            {
                "timestamp": int(row["timestamp"]),
                "itemid": int(row["itemid"]),
                "property_type": property_type,
                "property": property_id_mapping[(property_id, property_type)],
                "value": row["value"],
            }
        )

    with destination.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=["timestamp", "itemid", "property_type", "property", "value"])
        writer.writeheader()
        writer.writerows(integer_rows)

    return destination


def load_item_property_row_groups(
    item_properties_path: str | Path,
) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]], list[dict[str, int | str]]]:
    """Read item properties once and split rows into numeric, category, and non-numeric groups."""

    path = Path(item_properties_path).expanduser().resolve()

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
            property_type = row.get("property_type", "")
            raw_value = str(row["value"])
            item_property_key = (item_id, property_id)

            if str(property_type) == "0":
                grouped_category_rows.setdefault(item_property_key, []).append(
                    {
                        "timestamp": timestamp,
                        "value": raw_value,
                    }
                )
                continue

            if str(property_type) == "2":
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
                if not math.isfinite(mean_value) or not math.isfinite(std_value):
                    raise ValueError(
                        f"Non-finite numeric statistics for property {property_id}: "
                        f"mean={mean_value}, std={std_value}"
                    )
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
        if not math.isfinite(normalized_value):
            raise ValueError(
                f"Non-finite normalized value for property {property_id} on item {row['itemid']}: "
                f"raw={numeric_value}, mean={mean_value}, std={std_value}"
            )

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
    """Write merged non-numeric rows with timestamp-prefixed split token groups."""

    item_properties_path = Path(item_properties_path).expanduser().resolve()
    output_path = item_properties_path.with_name(output_filename)
    non_numeric_row_count = _merge_property_rows(
        rows=non_numeric_rows,
        output_path=output_path,
        value_columns=["num_values", "tokens"],
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

    events_time_range_outputs = generate_events_time_range(dataset_path)
    print(
        "Finished generate_events_time_range -> "
        f'{events_time_range_outputs["events_time_range_path"]}'
    )

    merged_events_outputs = merge_events_by_visitor_item(
        dataset_path,
        cutoff_date=str(events_time_range_outputs["cutoff_date"]),
    )
    print(
        "Finished merge_events_by_visitor_item -> "
        f'{merged_events_outputs["user_item_path"]} and '
        f'{merged_events_outputs["user_events_path"]} '
        f'(negative={merged_events_outputs["negative_data_count"]:,}, '
        f'positive={merged_events_outputs["positive_data_count"]:,}, '
        f'negative_cold_start={merged_events_outputs["negative_cold_start_data_count"]:,}, '
        f'positive_cold_start={merged_events_outputs["positive_cold_start_data_count"]:,})'
    )

    bucket_index_property_outputs = write_bucket_index_property_files(
        dataset_path,
        cutoff_date=str(merged_events_outputs["cutoff_date"]),
    )
    print(
        "Finished write_bucket_index_property_files -> "
        f'{bucket_index_property_outputs["numeric_properties_bucket_idx_path"]}, '
        f'{bucket_index_property_outputs["cate_properties_bucket_idx_path"]}, and '
        f'{bucket_index_property_outputs["non_numeric_properties_bucket_idx_path"]}'
    )

    return {
        "events_time_range_path": events_time_range_outputs["events_time_range_path"],
        "user_item_path": merged_events_outputs["user_item_path"],
        "user_events_path": merged_events_outputs["user_events_path"],
        "category_lookup_path": category_lookup_path,
        "leaf_depth_statistics": leaf_depth_statistics,
        "merged_item_properties_path": merged_item_properties_path,
        "property_id_map_path": merged_item_properties_path.with_name("property_id_map.csv"),
        "numeric_properties_path": numeric_property_outputs["numeric_properties_path"],
        "property_mean_std_path": numeric_property_outputs["property_mean_std_path"],
        "cate_properties_path": category_property_outputs["cate_properties_path"],
        "non_numeric_properties_path": non_numeric_property_outputs["non_numeric_properties_path"],
        "numeric_properties_bucket_idx_path": bucket_index_property_outputs["numeric_properties_bucket_idx_path"],
        "cate_properties_bucket_idx_path": bucket_index_property_outputs["cate_properties_bucket_idx_path"],
        "non_numeric_properties_bucket_idx_path": bucket_index_property_outputs["non_numeric_properties_bucket_idx_path"],
        "events_merged_path": merged_events_outputs["events_merged_path"],
    }


class UserItemEmbeddingIterableDataset(IterableDataset[tuple[Tensor, Tensor, Tensor]]):
    """Iterable dataset that samples user-item rows for training."""

    @classmethod
    def _from_rows(
        cls,
        *,
        user_item_path: Path,
        resources,
        token_transformer,
        item_projection,
        user_projection,
        device: torch.device,
        positive_negative_ratio: tuple[float, float] | None,
        samples_per_epoch: int | None,
        shuffle: bool,
        available_filter: int | str | None,
        is_cold_start_item_filter: int | str | None,
        positive_rows: tuple[tuple[int, int, int], ...],
        negative_rows: tuple[tuple[int, int, int], ...],
    ) -> "UserItemEmbeddingIterableDataset":
        dataset = cls.__new__(cls)
        super(UserItemEmbeddingIterableDataset, dataset).__init__()
        dataset.user_item_path = user_item_path
        dataset.resources = resources
        dataset.token_transformer = token_transformer
        dataset.item_projection = item_projection
        dataset.user_projection = user_projection
        dataset.device = torch.device(device)
        dataset.shuffle = shuffle
        dataset.positive_negative_ratio = positive_negative_ratio
        dataset.available_filter = None if available_filter is None else str(available_filter)
        dataset.is_cold_start_item_filter = (
            None if is_cold_start_item_filter is None else str(is_cold_start_item_filter)
        )
        dataset.positive_rows = positive_rows
        dataset.negative_rows = negative_rows
        dataset.all_rows = positive_rows + negative_rows
        if positive_negative_ratio is None:
            dataset.positive_sample_probability = None
        else:
            ratio_total = positive_negative_ratio[0] + positive_negative_ratio[1]
            dataset.positive_sample_probability = float(positive_negative_ratio[0] / ratio_total)
        dataset.sample_log_time = {"user": 0.0, "item": 0.0, "cnt": 0}
        dataset.samples_per_epoch = (
            len(dataset.all_rows)
            if samples_per_epoch is None
            else samples_per_epoch
        )
        if dataset.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be greater than 0.")
        if positive_negative_ratio is not None:
            if positive_negative_ratio[0] > 0 and not positive_rows:
                raise ValueError("No positive rows are available for the requested sampling ratio.")
            if positive_negative_ratio[1] > 0 and not negative_rows:
                raise ValueError("No negative rows are available for the requested sampling ratio.")
        elif dataset.samples_per_epoch > len(dataset.all_rows):
            raise ValueError(
                "samples_per_epoch cannot exceed the number of rows when positive_negative_ratio is None."
            )
        return dataset

    def __init__(
        self,
        user_item_path: str | Path,
        resources,
        token_transformer,
        item_projection,
        user_projection,
        device: str | torch.device,
        positive_negative_ratio: float | tuple[int | float, int | float] = 1.0,
        samples_per_epoch: int | None = None,
        shuffle: bool = True,
        available_filter: int | str | None = None,
        is_cold_start_item_filter: int | str | None = None,
    ) -> None:
        super().__init__()
        self.user_item_path = Path(user_item_path).expanduser().resolve()
        self.resources = resources
        self.token_transformer = token_transformer
        self.item_projection = item_projection
        self.user_projection = user_projection
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.available_filter = None if available_filter is None else str(available_filter)
        self.is_cold_start_item_filter = (
            None if is_cold_start_item_filter is None else str(is_cold_start_item_filter)
        )

        if isinstance(positive_negative_ratio, tuple):
            positive_ratio, negative_ratio = positive_negative_ratio
        else:
            positive_ratio, negative_ratio = float(positive_negative_ratio), 1.0

        if positive_ratio < 0 or negative_ratio < 0:
            raise ValueError("positive_negative_ratio values must be non-negative.")
        if positive_ratio == 0 and negative_ratio == 0:
            raise ValueError("positive_negative_ratio cannot be (0, 0).")
        normalized_ratio = (float(positive_ratio), float(negative_ratio))

        positive_rows: list[tuple[int, int, int]] = []
        negative_rows: list[tuple[int, int, int]] = []
        with self.user_item_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"visitorid", "itemid", "label"}
            missing_columns = required_columns.difference(reader.fieldnames or [])
            if missing_columns:
                missing_columns_str = ", ".join(sorted(missing_columns))
                raise ValueError(
                    f"{self.user_item_path} is missing required columns: {missing_columns_str}."
                )
            for row in reader:
                if (
                    self.available_filter is not None
                    and str(row.get("available", "")) != self.available_filter
                ):
                    continue
                if (
                    self.is_cold_start_item_filter is not None
                    and str(row.get("is_cold_start_item", "")) != self.is_cold_start_item_filter
                ):
                    continue
                label = int(row["label"])
                item_id = self.resources.item_id_by_original_id.get(int(row["itemid"]))
                if item_id is None:
                    continue
                sample = (int(row["visitorid"]), item_id, label)
                if label == 1:
                    positive_rows.append(sample)
                elif label == 0:
                    negative_rows.append(sample)
                else:
                    raise ValueError(f"Unsupported label value {label!r} in {self.user_item_path}.")

        if positive_ratio > 0 and not positive_rows:
            raise ValueError("No positive rows are available for the requested sampling ratio.")
        if negative_ratio > 0 and not negative_rows:
            raise ValueError("No negative rows are available for the requested sampling ratio.")

        self.positive_negative_ratio = normalized_ratio
        self.positive_rows = tuple(positive_rows)
        self.negative_rows = tuple(negative_rows)
        self.all_rows = self.positive_rows + self.negative_rows
        ratio_total = normalized_ratio[0] + normalized_ratio[1]
        self.positive_sample_probability = float(normalized_ratio[0] / ratio_total)
        self.samples_per_epoch = (
            len(self.positive_rows) + len(self.negative_rows)
            if samples_per_epoch is None
            else samples_per_epoch
        )
        self.sample_log_time = {"user": 0.0, "item": 0.0, "cnt": 0}
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be greater than 0.")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_row(
        self,
        rows: tuple[tuple[int, int, int], ...],
        rng: random.Random,
    ) -> tuple[int, int, int]:
        if not rows:
            raise ValueError("Cannot sample from an empty row pool.")
        return rows[rng.randrange(len(rows))]

    def _build_sample(
        self,
        visitor_id: int,
        item_id: int,
        label: int,
    ) -> tuple[Tensor, Tensor, Tensor]:

        user_embedding = get_user_embedding(
            visitor_id=visitor_id,
            resources=self.resources,
            token_transformer=self.token_transformer,
            user_projection=self.user_projection,
            device=self.device,
        )
        item_embedding = get_item_embedding(
            item_id=item_id,
            resources=self.resources,
            token_transformer=self.token_transformer,
            item_projection=self.item_projection,
            device=self.device,
        )
        label_tensor = torch.tensor(float(label), dtype=torch.float32, device=self.device)
        return user_embedding, item_embedding, label_tensor

    def split(
        self,
        train_size: float = 0.8,
        seed: int = 42,
        train_samples_per_epoch: int | None = None,
        test_samples_per_epoch: int | None = None,
    ) -> tuple["UserItemEmbeddingIterableDataset", "UserItemEmbeddingIterableDataset"]:
        if train_size <= 0 or train_size >= 1:
            raise ValueError("train_size must be between 0 and 1.")

        rng = random.Random(seed)
        positive_rows = list(self.positive_rows)
        negative_rows = list(self.negative_rows)
        rng.shuffle(positive_rows)
        rng.shuffle(negative_rows)

        positive_split_index = int(len(positive_rows) * train_size)
        negative_split_index = int(len(negative_rows) * train_size)

        train_dataset = self._from_rows(
            user_item_path=self.user_item_path,
            resources=self.resources,
            token_transformer=self.token_transformer,
            item_projection=self.item_projection,
            user_projection=self.user_projection,
            device=self.device,
            positive_negative_ratio=self.positive_negative_ratio,
            samples_per_epoch=train_samples_per_epoch,
            shuffle=self.shuffle,
            available_filter=self.available_filter,
            is_cold_start_item_filter=self.is_cold_start_item_filter,
            positive_rows=tuple(positive_rows[:positive_split_index]),
            negative_rows=tuple(negative_rows[:negative_split_index]),
        )
        test_dataset = self._from_rows(
            user_item_path=self.user_item_path,
            resources=self.resources,
            token_transformer=self.token_transformer,
            item_projection=self.item_projection,
            user_projection=self.user_projection,
            device=self.device,
            positive_negative_ratio=None,
            samples_per_epoch=test_samples_per_epoch,
            shuffle=False,
            available_filter=self.available_filter,
            is_cold_start_item_filter=self.is_cold_start_item_filter,
            positive_rows=tuple(positive_rows[positive_split_index:]),
            negative_rows=tuple(negative_rows[negative_split_index:]),
        )
        return train_dataset, test_dataset

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_sample_count = (self.samples_per_epoch + num_workers - 1) // num_workers
        rng = random.Random(time.time_ns() + worker_id)

        if self.positive_negative_ratio is None:
            if self.shuffle:
                row_pool = list(self.all_rows)
                rng.shuffle(row_pool)
            else:
                row_pool = self.all_rows

            for sample_index in range(worker_sample_count):
                global_index = sample_index * num_workers + worker_id
                if global_index >= self.samples_per_epoch:
                    break
                visitor_id, item_id, label = row_pool[global_index]
                yield self._build_sample(visitor_id=visitor_id, item_id=item_id, label=label)
            return

        for sample_index in range(worker_sample_count):
            global_index = sample_index * num_workers + worker_id
            if global_index >= self.samples_per_epoch:
                break

            if self.shuffle:
                label_to_draw = 1 if rng.random() < self.positive_sample_probability else 0
            else:
                cycle_position = global_index / max(self.samples_per_epoch, 1)
                label_to_draw = 1 if cycle_position < self.positive_sample_probability else 0
            if label_to_draw == 1:
                visitor_id, item_id, label = self._sample_row(self.positive_rows, rng)
            else:
                visitor_id, item_id, label = self._sample_row(self.negative_rows, rng)
            yield self._build_sample(visitor_id=visitor_id, item_id=item_id, label=label)


def collate_user_item_embedding_batch(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    if not batch:
        raise ValueError("batch must contain at least one sample.")
    user_embeddings, item_embeddings, labels = zip(*batch)
    return (
        torch.stack(user_embeddings, dim=0),
        torch.stack(item_embeddings, dim=0),
        torch.stack(labels, dim=0),
    )


__all__ = [
    "build_category_lookup_rows",
    "collate_user_item_embedding_batch",
    "compute_leaf_depth_statistics",
    "download_kaggle_dataset",
    "generate_events_time_range",
    "generate_category_lookup_table",
    "load_item_property_row_groups",
    "load_category_tree",
    "load_property_id_map",
    "merge_events_by_visitor_item",
    "process_category_property_values",
    "merge_item_properties_files",
    "process_non_numeric_property_values",
    "process_numeric_property_values",
    "preprocess_retailrocket_data",
    "resolve_dataset_file_path",
    "resolve_category_tree_path",
    "UserItemEmbeddingIterableDataset",
    "write_bucket_index_property_files",
]
