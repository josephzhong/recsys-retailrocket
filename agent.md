# Project Description

This project uses the [Retailrocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) to train and evaluate a recommender system model for e-commerce interactions.

The dataset is well suited to recommendation work because it contains implicit-feedback user behavior from a real-world online store. The main interaction log records three event types: `view`, `addtocart`, and `transaction`, which makes it useful for tasks such as session-based recommendation, next-item prediction, ranking, and conversion-oriented modeling.

# Dataset Summary

According to the Kaggle dataset page, the data is organized into three main parts:

- `events.csv`: user behavior events collected over about 4.5 months
- `item_properties.csv`: time-dependent item attributes such as category and availability, with property changes represented over time
- `category_tree.csv`: the product category hierarchy

The Kaggle page also describes the dataset as containing:

- 2,756,101 total events
- 2,664,312 views
- 69,332 add-to-cart events
- 22,457 transactions
- 1,407,580 unique visitors
- item properties for 417,053 unique items

The raw values are hashed for confidentiality, while some fields such as `categoryid` and `available` remain interpretable. This makes the dataset realistic for recommender-system experimentation while still protecting sensitive business information.

# Modeling Goal

The initial goal of this repository is to build a recommender system that learns from browsing and purchase behavior in the Retailrocket dataset. Over time, this project can expand to include preprocessing, feature engineering, candidate generation, ranking models, and evaluation notebooks built on top of the downloaded dataset.

# Preprocess Outputs

The `preprocess_retailrocket_data()` pipeline generates the following CSV files under the dataset directory.

## `category_lookup.csv`

Meaning: flattened category paths derived from `category_tree.csv`. Each row represents one category lineage from the root down to a leaf or intermediate category.

Columns:

- `level_1_category_id`: root category ID for the lineage.
- `level_2_category_id` to `level_6_category_id`: deeper category IDs in the same lineage. Blank when the lineage is shorter than that depth.

## `item_properties.csv`

Meaning: merged version of `item_properties_part1.csv` and `item_properties_part2.csv`, with property IDs remapped separately inside each property type.

Columns:

- `timestamp`: property change timestamp in milliseconds since Unix epoch.
- `itemid`: item ID.
- `property_type`: property type code. `0=category`, `1=non-numeric`, `2=numeric`.
- `property`: type-local mapped property ID.
- `value`: raw property value from the source file.

## `property_id_map.csv`

Meaning: lookup table from original property IDs to the remapped property IDs used in `item_properties.csv`, split by property type.

Columns:

- `original_property_id`: original property name or ID from the raw Retailrocket property files, such as `categoryid` or `available`.
- `property_type`: property type code. `0=category`, `1=non-numeric`, `2=numeric`.
- `mapped_property_id`: remapped property ID within that property type, starting from `0`.

## `numeric_properties.csv`

Meaning: item-level numeric properties after deduplication, time-history merging, and z-score normalization.

Columns:

- `itemid`: item ID.
- `property`: mapped numeric property ID.
- `value`: merged numeric history as `timestamp|normalized_value` entries joined by `+`.

## `property_mean_std.csv`

Meaning: normalization statistics for numeric properties.

Columns:

- `propertyid`: mapped numeric property ID.
- `original_value_count`: number of original numeric values observed for that property.
- `added_value_count`: number of synthetic duplicated values added for single-observation item-property groups before computing statistics.
- `total_value_count`: total values used to compute mean and standard deviation.
- `mean_value`: mean of the numeric values used for normalization.
- `std_value`: standard deviation of the numeric values used for normalization.

## `cate_properties.csv`

Meaning: item-level category-like properties after deduplication and time-history merging. This includes properties such as `categoryid` and `available`.

Columns:

- `itemid`: item ID.
- `property`: mapped category property ID.
- `value`: merged property history as `timestamp|value` entries joined by `+`.

## `non_numeric_properties.csv`

Meaning: item-level non-numeric properties after deduplication and tokenization into numeric-like tokens and text tokens.

Columns:

- `itemid`: item ID.
- `property`: mapped non-numeric property ID.
- `num_values`: merged history of extracted numeric-like tokens as `timestamp|value` entries joined by `+`.
- `tokens`: merged history of extracted non-numeric text tokens as `timestamp|value` entries joined by `+`.

## `events_time_range.csv`

Meaning: global event time range plus per-item event time ranges used to define the training cutoff and cold-start items.

Columns:

- `scope`: `global` for the dataset-wide summary row, `item` for per-item rows.
- `itemid`: item ID for `scope=item`; blank for the global row.
- `min_timestamp`: for the global row, the earliest event timestamp across all events; for item rows, the earliest `view` timestamp for that item.
- `max_timestamp`: latest event timestamp across all events for the global row, or latest event timestamp for the item row.
- `min_date`: UTC date version of `min_timestamp` in `YYYY-MM-DD`.
- `max_date`: UTC date version of `max_timestamp` in `YYYY-MM-DD`.
- `cutoff_date`: global cutoff date, defined as global `min_date + 4 months`.
- `is_cold_start_item`: for item rows, `1` if the item's first `view` date is on or after the global `cutoff_date`, else `0`. Blank for the global row.

## `events_merged.csv`

Meaning: visitor-item supervised samples built from `events.csv`, with pre-cutoff bucketed event histories, a label from post-cutoff views, item availability, and item cold-start status.

Columns:

- `visitorid`: visitor ID.
- `itemid`: item ID.
- `available`: item availability value inferred from the item's `available` property history at the cutoff.
- `by_user_item_view_event_list_str`: pre-cutoff view history for this visitor-item pair as `bucket_idx|count` entries joined by `+`.
- `by_user_item_cart_event_list_str`: pre-cutoff add-to-cart history for this visitor-item pair as `bucket_idx|count` entries joined by `+`.
- `by_user_item_transaction_event_list_str`: pre-cutoff transaction history for this visitor-item pair as `bucket_idx|count` entries joined by `+`.
- `label`: `1` if this visitor-item pair has at least one `view` event on or after the cutoff date, else `0`.
- `is_cold_start_item`: item-level cold-start flag loaded from `events_time_range.csv`.

Bucket definitions are relative to the global minimum event timestamp:

- `0`: `0` to `<1` day
- `1`: `1` to `<3` days
- `2`: `3` to `<7` days
- `3`: `7` to `<15` days
- `4`: `15` to `<30` days
- `5`: `>=30` days

## `user_events.csv`

Meaning: visitor-level aggregated histories built from all pre-cutoff visitor-item events, split by action type.

Columns:

- `visitorid`: visitor ID.
- `by_user_view_events_lists`: aggregated pre-cutoff view history as `itemid|bucket_idx|count` entries joined by `+`.
- `by_user_cart_events_lists`: aggregated pre-cutoff add-to-cart history as `itemid|bucket_idx|count` entries joined by `+`.
- `by_user_transaction_events_lists`: aggregated pre-cutoff transaction history as `itemid|bucket_idx|count` entries joined by `+`.
