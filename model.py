from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from tqdm import tqdm


PAD_TOKEN_ID = 0
UNKNOWN_TOKEN_ID = 1
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class NumericPropertyRecord:
    property_id: int
    values_by_bucket: dict[int, str]


@dataclass(frozen=True)
class NonNumericPropertyRecord:
    property_id: int
    num_values_by_bucket: dict[int, str]
    tokens_by_bucket: dict[int, str]


@dataclass(frozen=True)
class CategoryPropertyRecord:
    property_id: int
    values_by_bucket: dict[int, str]


@dataclass(frozen=True)
class ItemEmbeddingResources:
    numeric_properties_by_item: dict[int, tuple[NumericPropertyRecord, ...]]
    non_numeric_properties_by_item: dict[int, tuple[NonNumericPropertyRecord, ...]]
    category_properties_by_item: dict[int, tuple[CategoryPropertyRecord, ...]]
    numeric_vector_size: int
    category_vector_size: int
    category_ancestors_by_id: dict[int, tuple[int, ...]]
    category_index_by_id: dict[int, int]
    token_to_id: dict[str, int]
    item_id_by_original_id: dict[int, int]
    original_id_by_item_id: dict[int, int]
    non_numeric_item_property_index_by_key: dict[tuple[int, int], int]
    item_count: int
    item_id_map_path: Path
    item_property_key_map_path: Path


@dataclass(frozen=True)
class ItemEmbeddingModelState:
    resources: ItemEmbeddingResources
    token_transformer: "DecoderOnlyPropertyTransformer"
    item_projection: nn.Linear
    precomputed_non_numeric_token_embeddings: Tensor
    device: torch.device


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # Input/output shape: [batch_size, sequence_length, d_model].
        return x + self.pe[:, : x.size(1)]


class AttentionOnlyEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        attn_output, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(x + self.dropout(attn_output))


class DecoderOnlyPropertyTransformer(nn.Module):
    """A lightweight one-layer decoder-only transformer for non-numeric property tokens."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        nhead: int = 4,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.attention_layer = AttentionOnlyEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        self.dim_feedforward = dim_feedforward

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch_size, sequence_length].")

        # Maps token IDs [batch_size, sequence_length] to hidden states
        # [batch_size, sequence_length, d_model] with causal self-attention.
        token_ids = token_ids[:, : self.max_len]
        embeddings = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        embeddings = self.positional_encoding(embeddings)

        seq_len = token_ids.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=token_ids.device),
            diagonal=1,
        )
        padding_mask = token_ids.eq(PAD_TOKEN_ID)

        return self.attention_layer(
            embeddings,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
        )

    def get_next_token_latent_embedding(self, token_ids: Tensor) -> Tensor:
        # Returns one latent vector per sequence with shape [batch_size, d_model].
        hidden_states = self.forward(token_ids)
        valid_lengths = token_ids.ne(PAD_TOKEN_ID).sum(dim=1).clamp(min=1)
        batch_indices = torch.arange(token_ids.size(0), device=token_ids.device)
        return hidden_states[batch_indices, valid_lengths - 1]

def _resolve_dataset_dir(dataset_path: str | Path | None = None) -> Path:
    if dataset_path is None:
        return DEFAULT_DATASET_DIR
    resolved = Path(dataset_path).expanduser().resolve()
    return resolved.parent if resolved.is_file() else resolved


def _parse_bucket_history(history: str) -> dict[int, str]:
    parsed: dict[int, str] = {}
    if not history:
        return parsed

    for bucket_value in history.split("+"):
        if not bucket_value:
            continue
        bucket_str, value = bucket_value.split("|", maxsplit=1)
        parsed[int(bucket_str)] = value
    return parsed


def _try_parse_float(value: str) -> float | None:
    if value == "":
        return None

    normalized = value.strip()
    if normalized.lower().startswith("n"):
        normalized = normalized[1:]

    try:
        parsed = float(normalized)
    except ValueError:
        return None

    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_numeric_values(value: str) -> list[float]:
    parsed_values: list[float] = []
    if not value:
        return parsed_values

    for token in value.split():
        parsed = _try_parse_float(token)
        if parsed is not None:
            parsed_values.append(parsed)
    return parsed_values


def _build_token_vocab(non_numeric_path: Path) -> dict[str, int]:
    token_to_id = {"<pad>": PAD_TOKEN_ID, "<unk>": UNKNOWN_TOKEN_ID}
    next_token_id = 2

    with non_numeric_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            token_history = _parse_bucket_history(str(row["tokens"]))
            for token_string in token_history.values():
                for token in token_string.split():
                    if token not in token_to_id:
                        token_to_id[token] = next_token_id
                        next_token_id += 1

    return token_to_id


def _build_item_id_mapping(
    dataset_dir: Path,
    source_filename: str = "item_properties.csv",
    output_filename: str = "item_id_map.csv",
) -> tuple[dict[int, int], dict[int, int], Path]:
    source_path = dataset_dir / source_filename
    output_path = dataset_dir / output_filename

    original_item_ids: set[int] = set()
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            original_item_ids.add(int(row["itemid"]))

    sorted_item_ids = sorted(original_item_ids)
    item_id_by_original_id = {
        original_item_id: mapped_item_id
        for mapped_item_id, original_item_id in enumerate(sorted_item_ids)
    }
    original_id_by_item_id = {
        mapped_item_id: original_item_id
        for original_item_id, mapped_item_id in item_id_by_original_id.items()
    }

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["original_item_id", "mapped_item_id"],
        )
        writer.writeheader()
        for original_item_id in sorted_item_ids:
            writer.writerow(
                {
                    "original_item_id": original_item_id,
                    "mapped_item_id": item_id_by_original_id[original_item_id],
                }
            )

    return item_id_by_original_id, original_id_by_item_id, output_path


def _build_item_property_key_mapping(
    dataset_dir: Path,
    non_numeric_properties_by_item: dict[int, list[NonNumericPropertyRecord]],
    output_filename: str = "item_property_key_map.csv",
) -> tuple[dict[tuple[int, int], int], Path]:
    output_path = dataset_dir / output_filename
    item_property_keys = sorted(
        (item_id, record.property_id)
        for item_id, records in non_numeric_properties_by_item.items()
        for record in records
    )
    index_by_key = {
        key: mapped_index
        for mapped_index, key in enumerate(item_property_keys)
    }

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "property_id", "mapped_index"],
        )
        writer.writeheader()
        for item_id, property_id in item_property_keys:
            writer.writerow(
                {
                    "item_id": item_id,
                    "property_id": property_id,
                    "mapped_index": index_by_key[(item_id, property_id)],
                }
            )

    return index_by_key, output_path


def load_item_embedding_resources(dataset_path: str | Path | None = None) -> ItemEmbeddingResources:
    # Builds cached lookup tables used by the embedding function, including:
    # numeric_vector_size = max_numeric_property_id + 2
    # category_vector_size = number_of_category_lookup_rows + 1
    dataset_dir = _resolve_dataset_dir(dataset_path)
    numeric_path = dataset_dir / "numeric_properties_bucket_idx.csv"
    non_numeric_path = dataset_dir / "non_numeric_properties_bucket_idx.csv"
    category_path = dataset_dir / "cate_properties_bucket_idx.csv"
    category_lookup_path = dataset_dir / "category_lookup.csv"
    item_id_by_original_id, original_id_by_item_id, item_id_map_path = _build_item_id_mapping(dataset_dir)

    numeric_properties_by_item: dict[int, list[NumericPropertyRecord]] = {}
    max_numeric_property_id = -1
    with numeric_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            original_item_id = int(row["itemid"])
            item_id = item_id_by_original_id.get(original_item_id)
            if item_id is None:
                continue
            property_id = int(row["property"])
            max_numeric_property_id = max(max_numeric_property_id, property_id)
            numeric_properties_by_item.setdefault(item_id, []).append(
                NumericPropertyRecord(
                    property_id=property_id,
                    values_by_bucket=_parse_bucket_history(str(row["value"])),
                )
            )

    non_numeric_properties_by_item: dict[int, list[NonNumericPropertyRecord]] = {}
    with non_numeric_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            original_item_id = int(row["itemid"])
            item_id = item_id_by_original_id.get(original_item_id)
            if item_id is None:
                continue
            property_id = int(row["property"])
            non_numeric_properties_by_item.setdefault(item_id, []).append(
                NonNumericPropertyRecord(
                    property_id=property_id,
                    num_values_by_bucket=_parse_bucket_history(str(row["num_values"])),
                    tokens_by_bucket=_parse_bucket_history(str(row["tokens"])),
                )
            )

    category_properties_by_item: dict[int, list[CategoryPropertyRecord]] = {}
    with category_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            original_item_id = int(row["itemid"])
            item_id = item_id_by_original_id.get(original_item_id)
            if item_id is None:
                continue
            property_id = int(row["property"])
            category_properties_by_item.setdefault(item_id, []).append(
                CategoryPropertyRecord(
                    property_id=property_id,
                    values_by_bucket=_parse_bucket_history(str(row["value"])),
                )
            )

    category_ancestors_by_id: dict[int, tuple[int, ...]] = {}
    category_index_by_id: dict[int, int] = {}
    category_row_count = 0
    with category_lookup_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lineage = tuple(int(value) for value in row.values() if value not in {"", None})
            if not lineage:
                continue
            category_row_count += 1
            category_ancestors_by_id[lineage[-1]] = lineage
            category_index_by_id[lineage[-1]] = category_row_count

    non_numeric_item_property_index_by_key, item_property_key_map_path = _build_item_property_key_mapping(
        dataset_dir=dataset_dir,
        non_numeric_properties_by_item=non_numeric_properties_by_item,
    )

    return ItemEmbeddingResources(
        numeric_properties_by_item={
            item_id: tuple(records) for item_id, records in numeric_properties_by_item.items()
        },
        non_numeric_properties_by_item={
            item_id: tuple(records)
            for item_id, records in non_numeric_properties_by_item.items()
        },
        category_properties_by_item={
            item_id: tuple(records) for item_id, records in category_properties_by_item.items()
        },
        numeric_vector_size=max_numeric_property_id + 2,
        category_vector_size=category_row_count + 1,
        category_ancestors_by_id=category_ancestors_by_id,
        category_index_by_id=category_index_by_id,
        token_to_id=_build_token_vocab(non_numeric_path),
        item_id_by_original_id=item_id_by_original_id,
        original_id_by_item_id=original_id_by_item_id,
        non_numeric_item_property_index_by_key=non_numeric_item_property_index_by_key,
        item_count=len(item_id_by_original_id),
        item_id_map_path=item_id_map_path,
        item_property_key_map_path=item_property_key_map_path,
    )


def _build_token_transformer(
    resources: ItemEmbeddingResources,
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    dropout: float,
    max_len: int,
    device: str | torch.device,
) -> DecoderOnlyPropertyTransformer:
    transformer = DecoderOnlyPropertyTransformer(
        vocab_size=max(resources.token_to_id.values()) + 1,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
    )
    return transformer.to(torch.device(device))


def _precompute_non_numeric_token_embeddings(
    resources: ItemEmbeddingResources,
    token_transformer: DecoderOnlyPropertyTransformer,
    dataset_dir: Path,
    device: str | torch.device,
    batch_size: int = 256,
    load_if_exists: bool = True,
    save_if_built: bool = True,
    track_gradients: bool = False,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if track_gradients and load_if_exists:
        raise ValueError("track_gradients=True cannot be combined with load_if_exists=True.")
    if track_gradients and save_if_built:
        raise ValueError("track_gradients=True cannot be combined with save_if_built=True.")

    resolved_device = torch.device(device)
    destination = dataset_dir / f"item_property_bucket_token_embeddings_d{token_transformer.d_model}.pt"
    expected_pair_count = len(resources.non_numeric_item_property_index_by_key)

    if load_if_exists and destination.exists():
        cached_embeddings = torch.load(destination, map_location=resolved_device)
        if (
            cached_embeddings.ndim == 3
            and cached_embeddings.shape[0] == expected_pair_count
            and cached_embeddings.shape[1] == 6
            and cached_embeddings.shape[2] == token_transformer.d_model
        ):
            return cached_embeddings.to(resolved_device)

    item_property_records = sorted(
        (
            item_id,
            record.property_id,
            record,
            resources.non_numeric_item_property_index_by_key[(item_id, record.property_id)],
        )
        for item_id, records in resources.non_numeric_properties_by_item.items()
        for record in records
    )

    token_transformer = token_transformer.to(resolved_device)
    was_training = token_transformer.training
    if not track_gradients:
        token_transformer.eval()
    batch_embedding_chunks: list[Tensor] = []
    progress_bar = tqdm(
        range(0, len(item_property_records), batch_size),
        total=math.ceil(len(item_property_records) / batch_size) if item_property_records else 0,
        desc="Precomputing non-numeric token embeddings",
        unit="batch",
    )
    try:
        for batch_start in progress_bar:
            batch_records = item_property_records[batch_start : batch_start + batch_size]
            sequence_entries: list[tuple[int, int, list[int]]] = []
            max_sequence_length = 0

            for _, _, record, mapped_index in batch_records:
                for bucket_id in range(6):
                    token_string = record.tokens_by_bucket.get(bucket_id, "")
                    token_ids = [
                        resources.token_to_id.get(token, UNKNOWN_TOKEN_ID)
                        for token in token_string.split()
                        if token
                    ]
                    if not token_ids:
                        continue
                    sequence_entries.append((mapped_index, bucket_id, token_ids))
                    max_sequence_length = max(max_sequence_length, len(token_ids))

            if not sequence_entries:
                batch_embeddings = torch.zeros(
                    (len(batch_records), 6, token_transformer.d_model),
                    dtype=torch.float32,
                    device=resolved_device,
                )
                batch_embedding_chunks.append(batch_embeddings)
                continue

            padded_token_ids = torch.full(
                (len(sequence_entries), max_sequence_length),
                PAD_TOKEN_ID,
                dtype=torch.long,
                device=resolved_device,
            )
            for sequence_index, (_, _, token_ids) in enumerate(sequence_entries):
                padded_token_ids[sequence_index, : len(token_ids)] = torch.tensor(
                    token_ids,
                    dtype=torch.long,
                    device=resolved_device,
                )

            if track_gradients:
                latent_embeddings = token_transformer.get_next_token_latent_embedding(padded_token_ids)
            else:
                with torch.no_grad():
                    latent_embeddings = token_transformer.get_next_token_latent_embedding(padded_token_ids)

            latent_embedding_by_key_bucket = {
                (mapped_index, bucket_id): latent_embeddings[sequence_index]
                for sequence_index, (mapped_index, bucket_id, _) in enumerate(sequence_entries)
            }
            batch_row_embeddings: list[Tensor] = []
            for _, _, _, mapped_index in batch_records:
                bucket_embeddings: list[Tensor] = []
                for bucket_id in range(6):
                    bucket_embeddings.append(
                        latent_embedding_by_key_bucket.get(
                            (mapped_index, bucket_id),
                            torch.zeros(
                                token_transformer.d_model,
                                dtype=torch.float32,
                                device=resolved_device,
                            ),
                        )
                    )
                batch_row_embeddings.append(torch.stack(bucket_embeddings, dim=0))
            batch_embedding_chunks.append(torch.stack(batch_row_embeddings, dim=0))
    finally:
        progress_bar.close()
        token_transformer.train(was_training)
    if batch_embedding_chunks:
        precomputed_embeddings = torch.cat(batch_embedding_chunks, dim=0)
    else:
        precomputed_embeddings = torch.zeros(
            (expected_pair_count, 6, token_transformer.d_model),
            dtype=torch.float32,
            device=resolved_device,
        )
    if save_if_built:
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(precomputed_embeddings.detach().cpu(), destination)
    return precomputed_embeddings


def initialize_item_embedding_resources(
    dataset_path: str | Path | None = None,
    transformer_d_model: int = 32,
    transformer_nhead: int = 4,
    transformer_dim_feedforward: int = 64,
    transformer_dropout: float = 0.0,
    transformer_max_len: int = 128,
    item_embedding_size: int = 256,
    device: str | torch.device | None = None,
    non_numeric_token_batch_size: int = 256,
    load_precomputed_non_numeric_if_exists: bool = True,
    save_precomputed_non_numeric_if_built: bool = True,
    track_non_numeric_token_gradients: bool = False,
) -> ItemEmbeddingModelState:
    # Loads all CSV-backed resources once, writes item_id_map.csv, and initializes
    # the shared token transformer for the mapped zero-based model item IDs.
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    resources = load_item_embedding_resources(dataset_path)
    token_transformer = _build_token_transformer(
        resources=resources,
        d_model=transformer_d_model,
        nhead=transformer_nhead,
        dim_feedforward=transformer_dim_feedforward,
        dropout=transformer_dropout,
        max_len=transformer_max_len,
        device=resolved_device,
    )
    precomputed_non_numeric_token_embeddings = _precompute_non_numeric_token_embeddings(
        resources=resources,
        token_transformer=token_transformer,
        dataset_dir=_resolve_dataset_dir(dataset_path),
        device=resolved_device,
        batch_size=non_numeric_token_batch_size,
        load_if_exists=load_precomputed_non_numeric_if_exists,
        save_if_built=save_precomputed_non_numeric_if_built,
        track_gradients=track_non_numeric_token_gradients,
    )
    bucket_embedding_dim = (
        resources.numeric_vector_size
        + token_transformer.d_model
        + 3
        + resources.category_vector_size
    )
    item_projection = nn.Linear(bucket_embedding_dim * 6, item_embedding_size).to(resolved_device)
    return ItemEmbeddingModelState(
        resources=resources,
        token_transformer=token_transformer,
        item_projection=item_projection,
        precomputed_non_numeric_token_embeddings=precomputed_non_numeric_token_embeddings,
        device=resolved_device,
    )


def _build_numeric_item_vector(
    item_id: int,
    bucket_id: int,
    resources: ItemEmbeddingResources,
    device: torch.device,
) -> Tensor:
    # Output shape: [numeric_vector_size], where index 0 is the default/missing slot
    # and numeric property p is written to index p + 1.
    numeric_item_vector = torch.zeros(resources.numeric_vector_size, dtype=torch.float32, device=device)
    for record in resources.numeric_properties_by_item.get(item_id, ()):
        parsed_value = _try_parse_float(record.values_by_bucket.get(bucket_id, ""))
        if parsed_value is not None:
            numeric_item_vector[record.property_id + 1] = parsed_value
    return numeric_item_vector


def _build_non_numeric_item_token_embedding(
    item_id: int,
    bucket_id: int,
    resources: ItemEmbeddingResources,
    precomputed_non_numeric_token_embeddings: Tensor,
    device: torch.device,
) -> Tensor:
    # Output shape: [d_model]. Each item-property pair uses a precomputed
    # [6, d_model] cache indexed by mapped item-property key and bucket.
    property_embeddings: list[Tensor] = []
    for record in resources.non_numeric_properties_by_item.get(item_id, ()):
        if not record.tokens_by_bucket.get(bucket_id, "").strip():
            continue
        mapped_index = resources.non_numeric_item_property_index_by_key.get((item_id, record.property_id))
        if mapped_index is None:
            continue
        bucket_embedding = precomputed_non_numeric_token_embeddings[mapped_index, bucket_id]
        if bucket_embedding.device != device:
            bucket_embedding = bucket_embedding.to(device)
        property_embeddings.append(bucket_embedding)

    if not property_embeddings:
        return torch.zeros(
            precomputed_non_numeric_token_embeddings.shape[-1],
            dtype=torch.float32,
            device=device,
        )

    return torch.stack(property_embeddings, dim=0).mean(dim=0)


def _build_non_numeric_item_value_vector(
    item_id: int,
    bucket_id: int,
    resources: ItemEmbeddingResources,
    device: torch.device,
) -> Tensor:
    # Output shape: [3] in the order [max_of_max, min_of_min, mean_of_means].
    property_max_values: list[float] = []
    property_min_values: list[float] = []
    property_mean_values: list[float] = []

    for record in resources.non_numeric_properties_by_item.get(item_id, ()):
        numeric_values = _parse_numeric_values(record.num_values_by_bucket.get(bucket_id, ""))
        if not numeric_values:
            continue
        property_max_values.append(max(numeric_values))
        property_min_values.append(min(numeric_values))
        property_mean_values.append(sum(numeric_values) / len(numeric_values))

    if not property_max_values:
        return torch.zeros(3, dtype=torch.float32, device=device)

    # Output order follows the requested per-property summary dimensions: max, min, mean.
    return torch.tensor(
        [
            max(property_max_values),
            min(property_min_values),
            sum(property_mean_values) / len(property_mean_values),
        ],
        dtype=torch.float32,
        device=device,
    )


def _build_category_item_vector(
    item_id: int,
    bucket_id: int,
    resources: ItemEmbeddingResources,
    device: torch.device,
) -> Tensor:
    # Output shape: [category_vector_size], where index 0 is the default/missing slot
    # and each matched ancestor category is activated at its row_index + 0 offset.
    category_item_vector = torch.zeros(resources.category_vector_size, dtype=torch.float32, device=device)

    for record in resources.category_properties_by_item.get(item_id, ()):
        raw_value = record.values_by_bucket.get(bucket_id, "")
        category_value = _try_parse_float(raw_value)
        if category_value is None:
            continue

        for ancestor_id in resources.category_ancestors_by_id.get(int(category_value), ()):
            category_index = resources.category_index_by_id.get(ancestor_id)
            if category_index is not None:
                category_item_vector[category_index] = 1.0

    return category_item_vector


def get_item_embedding_by_item_bucket(
    item_id: int,
    bucket_id: int,
    resources: ItemEmbeddingResources,
    precomputed_non_numeric_token_embeddings: Tensor,
    device: str | torch.device,
    timings: dict[str, float] | None = None,
) -> Tensor:
    # Final output shape:
    # [numeric_vector_size + d_model + 3 + category_vector_size].
    # item_id is the mapped zero-based model item ID from resources.item_id_map_path.
    if bucket_id < 0 or bucket_id > 6:
        raise ValueError("bucket_id must be in the range [0, 6].")

    resolved_device = torch.device(device)

    start_time = time.perf_counter()
    numeric_item_vector = _build_numeric_item_vector(item_id, bucket_id, resources, resolved_device)
    if timings is not None:
        timings["numeric"] += time.perf_counter() - start_time

    start_time = time.perf_counter()
    non_numeric_item_token_embedding = _build_non_numeric_item_token_embedding(
        item_id=item_id,
        bucket_id=bucket_id,
        resources=resources,
        precomputed_non_numeric_token_embeddings=precomputed_non_numeric_token_embeddings,
        device=resolved_device,
    )
    if timings is not None:
        timings["non_numeric_token"] += time.perf_counter() - start_time

    start_time = time.perf_counter()
    non_numeric_item_value_vector = _build_non_numeric_item_value_vector(
        item_id=item_id,
        bucket_id=bucket_id,
        resources=resources,
        device=resolved_device,
    )
    if timings is not None:
        timings["non_numeric_value"] += time.perf_counter() - start_time

    start_time = time.perf_counter()
    category_item_vector = _build_category_item_vector(
        item_id=item_id,
        bucket_id=bucket_id,
        resources=resources,
        device=resolved_device,
    )
    if timings is not None:
        timings["category"] += time.perf_counter() - start_time

    return torch.cat(
        [
            numeric_item_vector,
            non_numeric_item_token_embedding,
            non_numeric_item_value_vector,
            category_item_vector,
        ],
        dim=0,
    )


def get_item_embedding(
    item_id: int,
    resources: ItemEmbeddingResources,
    precomputed_non_numeric_token_embeddings: Tensor,
    item_projection: nn.Linear,
    device: str | torch.device,
    return_timings: bool = False,
) -> Tensor | tuple[Tensor, dict[str, float]]:
    # Final output shape:
    # [item_embedding_size]. This concatenates the 6 pre-cutoff bucket embeddings
    # first, then projects the long item vector down with a linear layer.
    # item_id is the mapped zero-based model item ID from resources.item_id_map_path.
    timings = {
        "numeric": 0.0,
        "non_numeric_token": 0.0,
        "non_numeric_value": 0.0,
        "category": 0.0,
        "projection": 0.0,
        "total": 0.0,
    }
    total_start_time = time.perf_counter()
    bucket_embeddings = [
        get_item_embedding_by_item_bucket(
            item_id=item_id,
            bucket_id=bucket_id,
            resources=resources,
            precomputed_non_numeric_token_embeddings=precomputed_non_numeric_token_embeddings,
            device=device,
            timings=timings,
        )
        for bucket_id in range(6)
    ]
    concatenated_bucket_embeddings = torch.cat(bucket_embeddings, dim=0)
    resolved_device = torch.device(device)
    item_projection = item_projection.to(resolved_device)
    projection_start_time = time.perf_counter()
    item_embedding = item_projection(concatenated_bucket_embeddings)
    timings["projection"] += time.perf_counter() - projection_start_time
    timings["total"] = time.perf_counter() - total_start_time
    if return_timings:
        return item_embedding, timings
    return item_embedding


def get_all_items_embedding(
    dataset_path: str | Path | None = None,
    item_embedding_size: int = 256,
    output_path: str | Path | None = None,
    load_if_exists: bool = True,
    transformer_d_model: int = 8,
    transformer_nhead: int = 2,
    transformer_dim_feedforward: int = 64,
    transformer_dropout: float = 0.0,
    transformer_max_len: int = 128,
    device: str | torch.device | None = None,
    non_numeric_token_batch_size: int = 256,
    progress_log_interval: int = 100,
    use_cached_non_numeric_token_embeddings: bool | None = None,
) -> Tensor:
    # Output shape: [num_of_items, item_embedding_size]. This first tries to load
    # a saved tensor from disk. If it does not exist, it initializes resources and
    # shared modules once, then builds the table from get_item_embedding(...).
    dataset_dir = _resolve_dataset_dir(dataset_path)
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else dataset_dir / f"all_item_embeddings_{item_embedding_size}.pt"
    )
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if load_if_exists and destination.exists():
        return torch.load(destination, map_location=resolved_device)

    if progress_log_interval <= 0:
        raise ValueError("progress_log_interval must be greater than 0.")
    resolved_use_cached_non_numeric_token_embeddings = (
        not torch.is_grad_enabled()
        if use_cached_non_numeric_token_embeddings is None
        else use_cached_non_numeric_token_embeddings
    )

    state = initialize_item_embedding_resources(
        dataset_path=dataset_dir,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_dim_feedforward=transformer_dim_feedforward,
        transformer_dropout=transformer_dropout,
        transformer_max_len=transformer_max_len,
        item_embedding_size=item_embedding_size,
        device=resolved_device,
        non_numeric_token_batch_size=non_numeric_token_batch_size,
        load_precomputed_non_numeric_if_exists=resolved_use_cached_non_numeric_token_embeddings,
        save_precomputed_non_numeric_if_built=resolved_use_cached_non_numeric_token_embeddings,
        track_non_numeric_token_gradients=not resolved_use_cached_non_numeric_token_embeddings,
    )
    print("initialize_item_embedding_resources done", flush=True)
    item_embeddings: list[Tensor] = []
    timing_totals = {
        "numeric": 0.0,
        "non_numeric_token": 0.0,
        "non_numeric_value": 0.0,
        "category": 0.0,
        "projection": 0.0,
        "total": 0.0,
    }
    progress_bar = tqdm(
        range(state.resources.item_count),
        total=state.resources.item_count,
        desc="Building item embeddings",
        unit="item",
        miniters=progress_log_interval,
    )
    for processed_count, item_id in enumerate(progress_bar, start=1):
        item_embedding, item_timings = get_item_embedding(
            item_id=item_id,
            resources=state.resources,
            precomputed_non_numeric_token_embeddings=state.precomputed_non_numeric_token_embeddings,
            item_projection=state.item_projection,
            device=state.device,
            return_timings=True,
        )
        item_embeddings.append(item_embedding)
        for timing_name, timing_value in item_timings.items():
            timing_totals[timing_name] += timing_value
        if (
            processed_count % progress_log_interval == 0
            or processed_count == state.resources.item_count
        ):
            tqdm.write(
                "item "
                f"{item_id}: "
                f"numeric={item_timings['numeric'] * 1000:.2f}ms, "
                f"token={item_timings['non_numeric_token'] * 1000:.2f}ms, "
                f"value={item_timings['non_numeric_value'] * 1000:.2f}ms, "
                f"category={item_timings['category'] * 1000:.2f}ms, "
                f"projection={item_timings['projection'] * 1000:.2f}ms, "
                f"total={item_timings['total'] * 1000:.2f}ms"
            )
            progress_bar.set_postfix(
                {
                    "avg_num_ms": f"{timing_totals['numeric'] * 1000 / processed_count:.2f}",
                    "avg_tok_ms": f"{timing_totals['non_numeric_token'] * 1000 / processed_count:.2f}",
                    "avg_val_ms": f"{timing_totals['non_numeric_value'] * 1000 / processed_count:.2f}",
                    "avg_cat_ms": f"{timing_totals['category'] * 1000 / processed_count:.2f}",
                    "avg_proj_ms": f"{timing_totals['projection'] * 1000 / processed_count:.2f}",
                    "avg_total_ms": f"{timing_totals['total'] * 1000 / processed_count:.2f}",
                },
                refresh=False,
            )
    progress_bar.close()
    all_item_embeddings = torch.stack(item_embeddings, dim=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_item_embeddings, destination)
    return all_item_embeddings


__all__ = [
    "DecoderOnlyPropertyTransformer",
    "ItemEmbeddingModelState",
    "PositionalEncoding",
    "initialize_item_embedding_resources",
    "ItemEmbeddingResources",
    "get_all_items_embedding",
    "get_item_embedding",
    "get_item_embedding_by_item_bucket",
    "load_item_embedding_resources",
]
