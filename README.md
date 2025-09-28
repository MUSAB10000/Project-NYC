# Project-NYC · NYC Airbnb Dashboard

Interactive Streamlit app to **explore**, **analyze**, and **predict** NYC Airbnb listings.

- **Explore** supply, prices, and host behavior with rich charts  
- **Map** patterns across the city (no Mapbox token required)  
- **Predict** nightly price using an XGBoost model

> **Live app:** https://project-nyc-mqzcaiayv7vxpahbwnb4ig.streamlit.app/
> **Repo:** https://github.com/MUSAB10000/Project-NYC

---
## 👤 team members
- Rawan  Alsaffar
- Khaled Alzahrani
- Musab Aalbdullatif


## ✨ Features

- **Filters:** Borough, room type, price range, and availability  
- **Overview:** KPIs, quick data sample, numeric/categorical “describe”, price boxplot, top boroughs/room types, correlation heatmap  
- **Price Analysis:** Comparative bars, scatter relationships, heatmap (borough × room type), distributions, top listings  
- **Geospatial (MapLibre):  
- **Prediction:** Select a **borough** + minimal numeric inputs; location/host features are inferred from borough averages to match the trained model’s feature order

---

## 📂 Repository Structure
```bash
├── Advance_EDA_NYC.ipynb      # Jupyter notebook for extended EDA
├── app.py                     # Streamlit dashboard app
├── data.csv                   # Dataset (NYC Airbnb listings)
├── requirements.txt           # Python dependencies
├── xgboost_airbnb.pkl         # Trained XGBoost regression model
└── README.md                  # Project documentation
```
📊 Dataset Details

File: data.csv
Source:[ Airbnb NYC Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)

| Column                           | Description                                                 |
| -------------------------------- | ----------------------------------------------------------- |
| `id`                             | Unique listing ID                                           |
| `name`                           | Name of the listing                                         |
| `host_id`                        | Unique host ID                                              |
| `neighbourhood_group`            | Borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island) |
| `neighbourhood`                  | Specific neighborhood                                       |
| `latitude` / `longitude`         | Geolocation of the listing                                  |
| `room_type`                      | Room type (Entire home/apt, Private room, etc.)             |
| `price`                          | Price per night (USD)                                       |
| `minimum_nights`                 | Minimum stay required                                       |
| `number_of_reviews`              | Total number of reviews                                     |
| `reviews_per_month`              | Average reviews per month                                   |
| `calculated_host_listings_count` | Number of listings per host                                 |
| `availability_365`               | Availability in days per year                               |

