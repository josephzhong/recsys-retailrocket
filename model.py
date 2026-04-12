from __future__ import annotations

import csv
import heapq
import math
import multiprocessing
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


PAD_TOKEN_ID = 0
UNKNOWN_TOKEN_ID = 1
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_ITEM_BUCKET_EMBEDDING_CACHE_SIZE = 100000


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
class UserEventHistory:
    item_id: int
    bucket_id: int
    count: int


@dataclass(frozen=True)
class ItemEmbeddingResources:
    numeric_properties_by_item: dict[int, tuple[NumericPropertyRecord, ...]]
    non_numeric_properties_by_item: dict[int, tuple[NonNumericPropertyRecord, ...]]
    category_properties_by_item: dict[int, tuple[CategoryPropertyRecord, ...]]
    user_histories_by_visitor: dict[int, dict[str, tuple[UserEventHistory, ...]]]
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
    user_projection: nn.Linear
    device: torch.device


@dataclass(frozen=True)
class LoadedModelBundle:
    token_transformer: "DecoderOnlyPropertyTransformer"
    item_projection: nn.Linear
    user_projection: nn.Linear
    fm_model: "FactorizationMachines"
    device: torch.device
    resources: ItemEmbeddingResources | None = None


@dataclass
class _ItemBucketEmbeddingCacheEntry:
    embedding: Tensor
    query_count: int
    version: int


class _ItemBucketEmbeddingCache:
    def __init__(self, max_size: int = DEFAULT_ITEM_BUCKET_EMBEDDING_CACHE_SIZE) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0.")
        self.max_size = max_size
        self._lock = multiprocessing.RLock()
        self.total_query_count = 0
        self._entries: dict[tuple[int, int], _ItemBucketEmbeddingCacheEntry] = {}
        self._min_heap: list[tuple[int, int, tuple[int, int]]] = []
        self.miss = 0
        self.hit = 0

    def _life_point(self, query_count: int) -> int:
        return query_count - self.total_query_count

    def _push_heap_entry(self, key: tuple[int, int]) -> None:
        entry = self._entries[key]
        heapq.heappush(self._min_heap, (entry.query_count, entry.version, key))

    def get(self, key: tuple[int, int]) -> Tensor | None:
        with self._lock:
            self.total_query_count += 1
            entry = self._entries.get(key)
            if entry is None:
                self.miss += 1
                return None
            self.hit += 1
            entry.query_count += 1
            entry.version += 1
            self._push_heap_entry(key)
            return entry.embedding

    def put(self, key: tuple[int, int], embedding: Tensor) -> None:
        with self._lock:
            if key in self._entries:
                return

            if len(self._entries) >= self.max_size and not self._evict_one_negative_life_point():
                return

            self._entries[key] = _ItemBucketEmbeddingCacheEntry(
                embedding=embedding,
                query_count=0,
                version=0,
            )
            self._push_heap_entry(key)

    def _evict_one_negative_life_point(self) -> bool:
        while self._min_heap:
            cached_query_count, cached_version, key = heapq.heappop(self._min_heap)
            entry = self._entries.get(key)
            if entry is None:
                continue
            if entry.query_count != cached_query_count or entry.version != cached_version:
                continue
            if self._life_point(entry.query_count) >= 0:
                self._push_heap_entry(key)
                return False
            del self._entries[key]
            return True
        return False


_ITEM_BUCKET_EMBEDDING_CACHE = _ItemBucketEmbeddingCache()


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
        key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        query = x
        key = x if key_value is None else key_value
        value = key
        attn_output, _ = self.self_attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(query + self.dropout(attn_output))


class DecoderOnlyPropertyTransformer(nn.Module):
    """A lightweight one-layer decoder-only transformer for non-numeric property tokens."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        nhead: int = 4,
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

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch_size, sequence_length].")

        # Left-padded token IDs [batch_size, sequence_length] are mapped directly
        # to one latent vector per sequence with shape [batch_size, d_model].
        token_ids = token_ids[:, : self.max_len]
        embeddings = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        embeddings = self.positional_encoding(embeddings)

        padding_mask = token_ids.eq(PAD_TOKEN_ID)
        query = embeddings[:, -1:, :]

        latent_embedding = self.attention_layer(
            query,
            key_value=embeddings,
            key_padding_mask=padding_mask,
        )
        return latent_embedding.squeeze(1)

    def get_next_token_latent_embedding(self, token_ids: Tensor) -> Tensor:
        # Returns one latent vector per sequence with shape [batch_size, d_model].
        return self.forward(token_ids)


class FactorizationMachines(nn.Module):
    """Binary FM classifier over concatenated user and item embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int = 16,
        l2_reg_weight: float = 1e-6,
        dtype=torch.float32
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be greater than 0.")
        if l2_reg_weight < 0:
            raise ValueError("l2_reg_weight must be non-negative.")

        self.embedding_dim = embedding_dim
        self.input_dim = embedding_dim * 2
        self.latent_dim = latent_dim
        self.l2_reg_weight = l2_reg_weight
        self.linear = nn.Linear(self.input_dim, 1, dtype=dtype)
        self.factor_embeddings = nn.Parameter(
            torch.empty(self.input_dim, latent_dim, dtype=dtype)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.factor_embeddings)

    def _concatenate_features(
        self,
        user_embeddings: Tensor,
        item_embeddings: Tensor,
    ) -> Tensor:
        if user_embeddings.ndim != 2 or item_embeddings.ndim != 2:
            raise ValueError("user_embeddings and item_embeddings must both have shape [batch_size, embedding_dim].")
        if user_embeddings.shape != item_embeddings.shape:
            raise ValueError("user_embeddings and item_embeddings must have matching shapes.")
        if user_embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {user_embeddings.size(1)}."
            )
        return torch.cat((user_embeddings, item_embeddings), dim=1)

    def forward(
        self,
        user_embeddings: Tensor,
        item_embeddings: Tensor,
    ) -> Tensor:
        features = self._concatenate_features(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
        )
        features = torch.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        features = features.clamp(min=-100.0, max=100.0)
        linear_term = self.linear(features).squeeze(-1)
        projected_features = features @ self.factor_embeddings
        projected_squared = projected_features.square()
        squared_projected = features.square() @ self.factor_embeddings.square()
        pairwise_term = 0.5 * (projected_squared - squared_projected).sum(dim=1)
        return linear_term + pairwise_term

    def predict_proba(
        self,
        user_embeddings: Tensor,
        item_embeddings: Tensor,
    ) -> Tensor:
        return torch.sigmoid(self.forward(user_embeddings, item_embeddings))

    def compute_loss(
        self,
        user_embeddings: Tensor,
        item_embeddings: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = self.forward(user_embeddings, item_embeddings)
        labels = labels.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        if labels.shape != logits.shape:
            raise ValueError("labels must have shape [batch_size].")
        prediction_loss = self.loss_fn(logits, labels)
        l2_penalty = self.linear.weight.square().sum() + self.factor_embeddings.square().sum()
        if self.linear.bias is not None:
            l2_penalty = l2_penalty + self.linear.bias.square().sum()
        return logits, prediction_loss + self.l2_reg_weight * l2_penalty

    def forward_batch(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        user_embeddings, item_embeddings, labels = batch
        return self.compute_loss(user_embeddings, item_embeddings, labels)


def _get_transformer_config(
    token_transformer: DecoderOnlyPropertyTransformer,
) -> dict[str, int | float]:
    return {
        "vocab_size": token_transformer.token_embedding.num_embeddings,
        "d_model": token_transformer.d_model,
        "nhead": token_transformer.attention_layer.self_attn.num_heads,
        "dropout": float(token_transformer.attention_layer.self_attn.dropout),
        "max_len": token_transformer.max_len,
    }


def _get_fm_config(fm_model: FactorizationMachines) -> dict[str, int | float]:
    return {
        "embedding_dim": fm_model.embedding_dim,
        "latent_dim": fm_model.latent_dim,
        "l2_reg_weight": fm_model.l2_reg_weight,
    }


def save_model(
    path: str | Path,
    state: ItemEmbeddingModelState,
    fm_model: FactorizationMachines,
) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "transformer_config": _get_transformer_config(state.token_transformer),
        "item_projection_config": {
            "in_features": state.item_projection.in_features,
            "out_features": state.item_projection.out_features,
            "bias": state.item_projection.bias is not None,
        },
        "user_projection_config": {
            "in_features": state.user_projection.in_features,
            "out_features": state.user_projection.out_features,
            "bias": state.user_projection.bias is not None,
        },
        "fm_config": _get_fm_config(fm_model),
        "token_transformer_state_dict": state.token_transformer.state_dict(),
        "item_projection_state_dict": state.item_projection.state_dict(),
        "user_projection_state_dict": state.user_projection.state_dict(),
        "fm_state_dict": fm_model.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_model(
    path: str | Path,
    dataset_path: str | Path | None = None,
    device: str | torch.device | None = None,
    load_resources: bool = False,
) -> LoadedModelBundle:
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint at {checkpoint_path} must be a mapping.")

    transformer_config = checkpoint.get("transformer_config")
    item_projection_config = checkpoint.get("item_projection_config")
    user_projection_config = checkpoint.get("user_projection_config")
    fm_config = checkpoint.get("fm_config")
    if not isinstance(transformer_config, Mapping):
        raise ValueError("Checkpoint is missing transformer_config.")
    if not isinstance(item_projection_config, Mapping):
        raise ValueError("Checkpoint is missing item_projection_config.")
    if not isinstance(user_projection_config, Mapping):
        raise ValueError("Checkpoint is missing user_projection_config.")
    if not isinstance(fm_config, Mapping):
        raise ValueError("Checkpoint is missing fm_config.")

    token_transformer = DecoderOnlyPropertyTransformer(
        vocab_size=int(transformer_config["vocab_size"]),
        d_model=int(transformer_config["d_model"]),
        nhead=int(transformer_config["nhead"]),
        dropout=float(transformer_config["dropout"]),
        max_len=int(transformer_config["max_len"]),
    ).to(resolved_device)
    item_projection = nn.Linear(
        int(item_projection_config["in_features"]),
        int(item_projection_config["out_features"]),
        bias=bool(item_projection_config["bias"]),
    ).to(resolved_device)
    user_projection = nn.Linear(
        int(user_projection_config["in_features"]),
        int(user_projection_config["out_features"]),
        bias=bool(user_projection_config["bias"]),
    ).to(resolved_device)
    fm_model = FactorizationMachines(
        embedding_dim=int(fm_config["embedding_dim"]),
        latent_dim=int(fm_config["latent_dim"]),
        l2_reg_weight=float(fm_config["l2_reg_weight"]),
    ).to(resolved_device)

    token_transformer.load_state_dict(checkpoint["token_transformer_state_dict"])
    item_projection.load_state_dict(checkpoint["item_projection_state_dict"])
    user_projection.load_state_dict(checkpoint["user_projection_state_dict"])
    fm_model.load_state_dict(checkpoint["fm_state_dict"])

    resources = None
    if load_resources:
        if dataset_path is None:
            raise ValueError("dataset_path must be provided when load_resources=True.")
        resources = load_item_embedding_resources(dataset_path)

    return LoadedModelBundle(
        token_transformer=token_transformer,
        item_projection=item_projection,
        user_projection=user_projection,
        fm_model=fm_model,
        device=resolved_device,
        resources=resources,
    )

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


def _parse_user_event_history(
    history: str,
    item_id_by_original_id: dict[int, int],
) -> tuple[UserEventHistory, ...]:
    parsed_histories: list[UserEventHistory] = []
    if not history:
        return ()

    for raw_entry in history.split("+"):
        if not raw_entry:
            continue
        original_item_id_str, bucket_str, count_str = raw_entry.split("|", maxsplit=2)
        mapped_item_id = item_id_by_original_id.get(int(original_item_id_str))
        if mapped_item_id is None:
            continue

        bucket_id = int(bucket_str)
        if bucket_id < 0 or bucket_id >= 6:
            continue

        count = int(count_str)
        if count <= 0:
            continue

        parsed_histories.append(
            UserEventHistory(
                item_id=mapped_item_id,
                bucket_id=bucket_id,
                count=count,
            )
        )

    return tuple(parsed_histories)


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


def _signed_log1p_tensor(values: Tensor) -> Tensor:
    """Compress large-magnitude numeric summaries while preserving sign."""

    return values.sign() * torch.log1p(values.abs())


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

    user_events_path = dataset_dir / "user_events.csv"
    user_histories_by_visitor: dict[int, dict[str, tuple[UserEventHistory, ...]]] = {}
    with user_events_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            visitor_id = int(row["visitorid"])
            user_histories_by_visitor[visitor_id] = {
                "view": _parse_user_event_history(
                    str(row["by_user_view_events_lists"]),
                    item_id_by_original_id,
                ),
                "cart": _parse_user_event_history(
                    str(row["by_user_cart_events_lists"]),
                    item_id_by_original_id,
                ),
                "transaction": _parse_user_event_history(
                    str(row["by_user_transaction_events_lists"]),
                    item_id_by_original_id,
                ),
            }

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
        user_histories_by_visitor=user_histories_by_visitor,
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
    dropout: float,
    max_len: int,
    device: str | torch.device,
) -> DecoderOnlyPropertyTransformer:
    transformer = DecoderOnlyPropertyTransformer(
        vocab_size=max(resources.token_to_id.values()) + 1,
        d_model=d_model,
        nhead=nhead,
        dropout=dropout,
        max_len=max_len,
    )
    return transformer.to(torch.device(device))


def _encode_non_numeric_token_sequences(
    token_strings: list[str],
    resources: ItemEmbeddingResources,
    token_transformer: DecoderOnlyPropertyTransformer,
    device: torch.device,
) -> Tensor:
    if not token_strings:
        return torch.zeros((0, token_transformer.d_model), dtype=torch.float32, device=device)

    token_id_sequences = [
        [resources.token_to_id.get(token, UNKNOWN_TOKEN_ID) for token in token_string.split() if token]
        for token_string in token_strings
    ]
    token_id_sequences = [token_ids for token_ids in token_id_sequences if token_ids]
    if not token_id_sequences:
        return torch.zeros((0, token_transformer.d_model), dtype=torch.float32, device=device)

    max_sequence_length = max(len(token_ids) for token_ids in token_id_sequences)
    padded_token_ids = torch.full(
        (len(token_id_sequences), max_sequence_length),
        PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    for sequence_index, token_ids in enumerate(token_id_sequences):
        padded_token_ids[sequence_index, -len(token_ids) :] = padded_token_ids.new_tensor(token_ids)
    return token_transformer.get_next_token_latent_embedding(padded_token_ids)


def initialize_item_embedding_resources(
    dataset_path: str | Path | None = None,
    transformer_d_model: int = 16,
    transformer_nhead: int = 2,
    transformer_dropout: float = 0.0,
    transformer_max_len: int = 32,
    item_embedding_size: int = 64,
    device: str | torch.device | None = None,
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
        dropout=transformer_dropout,
        max_len=transformer_max_len,
        device=resolved_device,
    )
    bucket_embedding_dim = (
        resources.numeric_vector_size
        + token_transformer.d_model
        + 3
        + resources.category_vector_size
    )
    item_projection = nn.Linear(bucket_embedding_dim * 6, item_embedding_size).to(resolved_device)
    user_projection = nn.Linear(bucket_embedding_dim * 6 * 3, item_embedding_size).to(resolved_device)
    torch.nn.init.xavier_uniform_(item_projection.weight)
    torch.nn.init.xavier_uniform_(user_projection.weight)
    return ItemEmbeddingModelState(
        resources=resources,
        token_transformer=token_transformer,
        item_projection=item_projection,
        user_projection=user_projection,
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
    token_transformer: DecoderOnlyPropertyTransformer,
    device: torch.device,
) -> Tensor:
    # Output shape: [d_model]. Each non-empty property token sequence for the
    # current item bucket is encoded on demand and then averaged.
    token_strings: list[str] = []
    for record in resources.non_numeric_properties_by_item.get(item_id, ()):
        token_string = record.tokens_by_bucket.get(bucket_id, "").strip()
        if token_string:
            token_strings.append(token_string)

    latent_embeddings = _encode_non_numeric_token_sequences(
        token_strings=token_strings,
        resources=resources,
        token_transformer=token_transformer,
        device=device,
    )
    if latent_embeddings.numel() == 0:
        return torch.zeros(token_transformer.d_model, dtype=torch.float32, device=device)
    return latent_embeddings.mean(dim=0)


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
        property_mean_values.append(sum(numeric_values) / len(numeric_values) if len(numeric_values) > 0 else 0.0 )

    if not property_max_values:
        return torch.zeros(3, dtype=torch.float32, device=device)

    # Output order follows the requested per-property summary dimensions: max, min, mean.
    raw_summary = torch.tensor(
        [
            max(property_max_values),
            min(property_min_values),
            sum(property_mean_values) / len(property_mean_values) if len(property_mean_values) > 0 else 0.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    return _signed_log1p_tensor(raw_summary)


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
    token_transformer: DecoderOnlyPropertyTransformer,
    device: str | torch.device,
    timings: dict[str, float] | None = None,
) -> Tensor:
    # Final output shape:
    # [numeric_vector_size + d_model + 3 + category_vector_size].
    # item_id is the mapped zero-based model item ID from resources.item_id_map_path.
    if bucket_id < 0 or bucket_id > 6:
        raise ValueError("bucket_id must be in the range [0, 6].")

    resolved_device = torch.device(device)
    cache_lookup_key = (item_id, bucket_id)
    cached_embedding = _ITEM_BUCKET_EMBEDDING_CACHE.get(cache_lookup_key)
    if cached_embedding is not None:
        return cached_embedding

    start_time = time.perf_counter()
    numeric_item_vector = _build_numeric_item_vector(item_id, bucket_id, resources, resolved_device)
    if timings is not None:
        timings["numeric"] += time.perf_counter() - start_time

    start_time = time.perf_counter()
    non_numeric_item_token_embedding = _build_non_numeric_item_token_embedding(
        item_id=item_id,
        bucket_id=bucket_id,
        resources=resources,
        token_transformer=token_transformer,
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

    item_bucket_embedding = torch.cat(
        [
            numeric_item_vector,
            non_numeric_item_token_embedding,
            non_numeric_item_value_vector,
            category_item_vector,
        ],
        dim=0,
    )
    _ITEM_BUCKET_EMBEDDING_CACHE.put(cache_lookup_key, item_bucket_embedding.detach())
    return item_bucket_embedding


def get_item_embedding(
    item_id: int,
    resources: ItemEmbeddingResources,
    token_transformer: DecoderOnlyPropertyTransformer,
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
            token_transformer=token_transformer,
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


def _build_event_type_bucket_embeddings(
    visitor_id: int,
    event_type: str,
    resources: ItemEmbeddingResources,
    token_transformer: DecoderOnlyPropertyTransformer,
    device: torch.device,
) -> Tensor:
    bucket_embedding_dim = (
        resources.numeric_vector_size
        + token_transformer.d_model
        + 3
        + resources.category_vector_size
    )
    bucket_embeddings = torch.zeros((6, bucket_embedding_dim), dtype=torch.float32, device=device)
    bucket_counts = torch.zeros(6, dtype=torch.float32, device=device)

    visitor_histories = resources.user_histories_by_visitor.get(visitor_id, {})
    for history in visitor_histories.get(event_type, ()):
        item_bucket_embedding = get_item_embedding_by_item_bucket(
            item_id=history.item_id,
            bucket_id=history.bucket_id,
            resources=resources,
            token_transformer=token_transformer,
            device=device,
        )
        bucket_embeddings[history.bucket_id] += item_bucket_embedding * float(history.count)
        bucket_counts[history.bucket_id] += float(history.count)

    non_empty_mask = bucket_counts > 0
    if non_empty_mask.any():
        bucket_embeddings[non_empty_mask] = torch.nan_to_num (
            (bucket_embeddings[non_empty_mask] / bucket_counts[non_empty_mask].unsqueeze(1)), 
            nan=0.0, posinf=100.0, neginf=-100.0
        )
    return bucket_embeddings


def get_user_embedding(
    visitor_id: int,
    resources: ItemEmbeddingResources,
    token_transformer: DecoderOnlyPropertyTransformer,
    user_projection: nn.Linear,
    device: str | torch.device,
) -> Tensor:
    # Final output shape:
    # [item_embedding_size]. For each of view/cart/transaction, this averages
    # per-bucket item-bucket embeddings over the visitor's events, concatenates
    # the 6 bucket vectors, then projects the 18-bucket history down.
    resolved_device = torch.device(device)
    event_bucket_embeddings = [
        _build_event_type_bucket_embeddings(
            visitor_id=visitor_id,
            event_type=event_type,
            resources=resources,
            token_transformer=token_transformer,
            device=resolved_device,
        )
        for event_type in ("view", "cart", "transaction")
    ]
    concatenated_event_history = torch.cat(event_bucket_embeddings, dim=0).reshape(-1)
    user_projection = user_projection.to(resolved_device)
    return user_projection(concatenated_event_history)
__all__ = [
    "FactorizationMachines",
    "DecoderOnlyPropertyTransformer",
    "ItemEmbeddingModelState",
    "LoadedModelBundle",
    "PositionalEncoding",
    "initialize_item_embedding_resources",
    "ItemEmbeddingResources",
    "get_item_embedding",
    "get_item_embedding_by_item_bucket",
    "get_user_embedding",
    "save_model",
    "load_model",
    "load_item_embedding_resources",
]
