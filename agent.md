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
