# NYC Airbnb Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns                
import matplotlib.pyplot as plt       
import joblib
from xgboost import XGBRegressor  

# 0) PAGE CONFIG
st.set_page_config(page_title="NYC Airbnb Dashboard", layout="wide")


# 1) LOAD DATA
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Read the dataset once and cache it for fast reruns."""
    return pd.read_csv("data.csv")

df = load_data()

# Sanity check: make sure required columns exist (fail early with a clear error)
required_cols = {
    "id", "host_id", "name", "neighbourhood_group", "neighbourhood",
    "latitude", "longitude", "room_type", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Your dataset is missing required columns: {sorted(list(missing))}")
    st.stop()

# Convert important columns to numeric and downcast floats to save memory (helps on Streamlit Cloud)
num_cols = [
    "latitude", "longitude", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[["latitude", "longitude", "price", "reviews_per_month"]] = (
    df[["latitude", "longitude", "price", "reviews_per_month"]].astype("float32")
)
for c in ["minimum_nights", "number_of_reviews", "calculated_host_listings_count", "availability_365"]:
    df[c] = df[c].astype("float32")

# 2) SIDEBAR FILTERS
st.sidebar.header("🔎 Filters")

# All available filter options (clean & sorted)
boroughs_all = sorted(df["neighbourhood_group"].dropna().unique())
room_types_all = sorted(df["room_type"].dropna().unique())

# Widgets
boroughs = st.sidebar.multiselect("Select Boroughs", boroughs_all, default=boroughs_all)
room_types = st.sidebar.multiselect("Select Room Types", room_types_all, default=room_types_all)

# Robust price slider bounds (ignore extreme outliers)
p99 = float(df["price"].quantile(0.99))
slider_max = int(max(50, min(p99 + 1, float(df["price"].max(skipna=True) or 1000))))
slider_min = int(max(0, float(df["price"].min(skipna=True) or 0)))

price_range = st.sidebar.slider("Price Range ($)", slider_min, slider_max, (slider_min, slider_max))
availability_range = st.sidebar.slider("Availability (days)", 0, 365, (0, 365))

# Performance toggle for maps
speed_mode = st.sidebar.toggle(
    "⚡ Speed mode (recommended)",
    value=True,
    help="Samples fewer points on maps to prevent memory issues.",
)

# Apply filters
df_filtered = df[
    df["neighbourhood_group"].isin(boroughs)
    & df["room_type"].isin(room_types)
    & df["price"].between(price_range[0], price_range[1], inclusive="both")
    & df["availability_365"].between(availability_range[0], availability_range[1], inclusive="both")
].copy()

if df_filtered.empty:
    st.error("No data matches the selected filters. Please adjust the sidebar filters.")
    st.stop()

# 3) NAVIGATION
tab = st.sidebar.radio(
    "📌 Go to Section:",
    ["Overview", "Price Analysis", "Geospatial Analysis", "Hosts & Reviews", "Prediction"],
)

st.title("🏠 NYC Airbnb Dashboard")
st.markdown("Explore, analyze, and predict New York City Airbnb listings with interactive visuals.")


# 4) OVERVIEW — KPIs + SAMPLE + DESCRIBE + BOX + TOP COUNTS
if tab == "Overview":
    st.header("📊 Dataset Overview")

    # --- KPIs on filtered data ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Avg Price ($)", f"{float(df_filtered['price'].mean()):.2f}")
    with k2:
        st.metric("Avg Minimum Nights", f"{float(df_filtered['minimum_nights'].mean()):.1f}")
    with k3:
        st.metric("Avg Reviews/Month", f"{float(df_filtered['reviews_per_month'].fillna(0).mean()):.2f}")
    with k4:
        st.metric("Avg Host Listings", f"{float(df_filtered['calculated_host_listings_count'].mean()):.1f}")

    # Quick look at data 
    st.subheader("🔎 Quick Look at the Data")
    st.caption("A small sample of the filtered dataset:")
    st.dataframe(df_filtered.sample(min(10, len(df_filtered)), random_state=42))

    with st.expander("📐 Descriptive Statistics (Numeric)"):
        st.dataframe(df_filtered.select_dtypes(include=np.number).describe().round(2))

    with st.expander("🔤 Descriptive Statistics (Categorical)"):
        cat_cols = df_filtered.select_dtypes(exclude=np.number).columns.tolist()
        st.dataframe(df_filtered[cat_cols].describe().T if cat_cols else pd.DataFrame())

    #Price boxplot (quick distribution check) 
    st.subheader("Price Distribution (Boxplot)")
    st.plotly_chart(px.box(df_filtered, y="price", points="outliers"), use_container_width=True)

    # Top boroughs by listings 
    st.subheader("Top Boroughs by Listings")
    borough_counts = (
        df_filtered["neighbourhood_group"]
        .value_counts()
        .rename_axis("Borough")
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    fig = px.bar(
        borough_counts,
        x="Borough",
        y="Count",
        color="Borough",
        text="Count",
        title="Listings per Borough (Descending Order)",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    #  Top room types by listings 
    st.subheader("Top Room Types")
    room_counts = (
        df_filtered["room_type"]
        .value_counts()
        .rename_axis("Room Type")
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    fig = px.bar(
        room_counts,
        x="Room Type",
        y="Count",
        color="Room Type",
        text="Count",
        title="Listings per Room Type (Descending Order)",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap 
    st.subheader("Correlation Heatmap")
    numerical_df = df_filtered.select_dtypes(include="number").drop(
        columns=["id", "host_id", "latitude", "longitude"], errors="ignore"
    )
    if numerical_df.shape[1] >= 2:
        corr = numerical_df.corr(numeric_only=True)
        fig_hm, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="viridis", annot=True, fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title("Correlation Heatmap of Numerical Features")
        st.pyplot(fig_hm)
    else:
        st.info("Not enough numeric features to compute a correlation heatmap.")

# 5) PRICE ANALYSIS — classic bar/scatters
elif tab == "Price Analysis":
    st.header("💰 Price Analysis")

    # 1) Avg price by borough
    st.subheader("1. Average Price by Borough")
    avg_price_borough = (
        df_filtered.groupby("neighbourhood_group", as_index=False)["price"]
        .mean()
        .sort_values(by="price", ascending=False)
    )
    fig = px.bar(
        avg_price_borough,
        x="neighbourhood_group",
        y="price",
        color="neighbourhood_group",
        text="price",
        title="Average Price by Borough",
    )
    fig.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 2) Avg price by room type
    st.subheader("2. Average Price by Room Type")
    avg_price_room = (
        df_filtered.groupby("room_type", as_index=False)["price"]
        .mean()
        .sort_values(by="price", ascending=False)
    )
    fig = px.bar(
        avg_price_room,
        x="room_type",
        y="price",
        color="room_type",
        text="price",
        title="Average Price by Room Type",
    )
    fig.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 3) Price vs minimum nights
    st.subheader("3. Price vs Minimum Nights (Scatter)")
    st.plotly_chart(
        px.scatter(
            df_filtered,
            x="minimum_nights",
            y="price",
            color="neighbourhood_group",
            log_y=True,
            title="Price vs Minimum Nights by Borough",
            hover_data=["name", "room_type"],
        ),
        use_container_width=True,
    )

    # 4) Price vs availability
    st.subheader("4. Price vs Availability (Scatter)")
    st.plotly_chart(
        px.scatter(
            df_filtered,
            x="availability_365",
            y="price",
            color="room_type",
            log_y=True,
            title="Price vs Availability by Room Type",
            hover_data=["name", "neighbourhood_group"],
        ),
        use_container_width=True,
    )

    # 5) Avg minimum nights by borough
    st.subheader("5. Average Minimum Nights by Borough")
    avg_min_nights_borough = (
        df_filtered.groupby("neighbourhood_group", as_index=False)["minimum_nights"]
        .mean()
        .sort_values(by="minimum_nights", ascending=False)
    )
    fig = px.bar(
        avg_min_nights_borough,
        x="neighbourhood_group",
        y="minimum_nights",
        color="neighbourhood_group",
        text="minimum_nights",
        title="Average Minimum Nights by Borough",
    )
    fig.update_traces(texttemplate="%{text:.1f} nights", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # 6) Mean price heatmap (borough × room type)
    st.subheader("6. Mean Price by Borough and Room Type (Heatmap)")
    pivot_table = df_filtered.groupby(["neighbourhood_group", "room_type"])["price"].mean().unstack()
    st.plotly_chart(
        px.imshow(
            pivot_table,
            labels=dict(x="Room Type", y="Borough", color="Mean Price"),
            x=pivot_table.columns,
            y=pivot_table.index,
            color_continuous_scale="YlGnBu",
            title="Mean Price by Borough and Room Type",
        ),
        use_container_width=True,
    )

    # 7) Price histogram
    st.subheader("7. Price Distribution (Histogram)")
    st.plotly_chart(
        px.histogram(
            df_filtered,
            x="price",
            nbins=50,
            color="room_type",
            title="Price Distribution by Room Type",
            marginal="box",
            log_y=True,
        ),
        use_container_width=True,
    )
# 6) GEOSPATIAL ANALYSIS — 7 memory-safe MapLibre charts

elif tab == "Geospatial Analysis":
    st.header("🌍 Geospatial Analysis (MapLibre)")

    # Keep only rows with valid coordinates and plausible NYC bounds
    geo_df = df_filtered.dropna(subset=["latitude", "longitude"]).copy()
    geo_df = geo_df[
        geo_df["latitude"].between(40.4, 41.1) & geo_df["longitude"].between(-74.3, -73.6)
    ]
    if geo_df.empty:
        st.warning("No valid latitude/longitude points after filtering. Try widening filters.")
        st.stop()

    # Limit map size to avoid MemoryError during Plotly JSON serialization
    MAX_POINTS = 5_000 if speed_mode else 12_000
    df_sample = geo_df.sample(min(MAX_POINTS, len(geo_df)), random_state=42).copy()

    # Compact hover payload to save memory
    base_hover = {"latitude": False, "longitude": False}
    hover_scatter = {**base_hover, "price": True, "room_type": True, "neighbourhood": True}
    hover_scatter_boro = {**base_hover, "price": True, "neighbourhood_group": True, "neighbourhood": True}

    # 1) Listings scatter
    st.subheader("1. Listings Map (Scatter)")
    st.plotly_chart(
        px.scatter_map(
            df_sample, lat="latitude", lon="longitude",
            color="neighbourhood_group", hover_name="name",
            hover_data=hover_scatter, zoom=9, height=440, map_style="open-street-map"
        ),
        use_container_width=True,
    )

    # 2) Price hotspots (density)
    st.subheader("2. Price Hotspots (Density)")
    density_df = df_sample.dropna(subset=["price"]).copy()
    if density_df.empty:
        st.info("No price values available for density map after filtering.")
    else:
        p = density_df["price"].clip(lower=0)
        w = (p - p.min()) / (p.max() - p.min()) if p.max() > p.min() else p * 0 + 0.5
        st.plotly_chart(
            px.density_map(
                density_df, lat="latitude", lon="longitude", z=w,
                radius=12, center={"lat": 40.73, "lon": -73.93},
                zoom=9, height=440, map_style="open-street-map"
            ),
            use_container_width=True,
        )

    # 3) Room type colored map
    st.subheader("3. Room Type Colored Map")
    st.plotly_chart(
        px.scatter_map(
            df_sample, lat="latitude", lon="longitude",
            color="room_type", hover_name="name",
            hover_data=hover_scatter_boro, zoom=9, height=440, map_style="open-street-map"
        ),
        use_container_width=True,
    )

    # 4) Price quantile map (discrete bands)
    st.subheader("4. Price Quantile Map")
    df_q = df_sample.dropna(subset=["price"]).copy()
    if df_q.empty:
        st.info("No prices available to compute quantiles.")
    else:
        df_q["price_quantile"] = pd.qcut(
            df_q["price"].rank(method="first"),
            q=5, labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"],
        )
        st.plotly_chart(
            px.scatter_map(
                df_q, lat="latitude", lon="longitude",
                color="price_quantile", hover_name="name",
                hover_data=hover_scatter_boro, zoom=9, height=440, map_style="open-street-map"
            ),
            use_container_width=True,
        )

    # 5) Neighborhood bubble map (size=count, color=avg price)
    st.subheader("5. Neighborhood Mean-Price Bubble Map")
    neigh_stats = (
        geo_df.groupby("neighbourhood", as_index=False)
        .agg(
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
            avg_price=("price", "mean"),
            listing_count=("id", "count"),
        )
    )
    st.plotly_chart(
        px.scatter_map(
            neigh_stats, lat="lat", lon="lon",
            size="listing_count", color="avg_price",
            hover_name="neighbourhood",
            hover_data={"avg_price": ":.2f", "listing_count": True},
            zoom=9, height=440, map_style="open-street-map"
        ),
        use_container_width=True,
    )

    # 6) Availability density
    st.subheader("6. Availability Density (days)")
    avail_df = df_sample.dropna(subset=["availability_365"]).copy()
    if avail_df.empty:
        st.info("No availability values available for density map.")
    else:
        a = avail_df["availability_365"].clip(lower=0)
        w = (a - a.min()) / (a.max() - a.min()) if a.max() > a.min() else a * 0 + 0.5
        st.plotly_chart(
            px.density_map(
                avail_df, lat="latitude", lon="longitude", z=w,
                radius=12, center={"lat": 40.73, "lon": -73.93},
                zoom=9, height=440, map_style="open-street-map"
            ),
            use_container_width=True,
        )

    # 7) Minimum nights density
    st.subheader("7. Minimum Nights Density")
    mn_df = df_sample.dropna(subset=["minimum_nights"]).copy()
    if mn_df.empty:
        st.info("No minimum_nights values available for density map.")
    else:
        m = mn_df["minimum_nights"].clip(lower=0)
        w = (m - m.min()) / (m.max() - m.min()) if m.max() > m.min() else m * 0 + 0.5
        st.plotly_chart(
            px.density_map(
                mn_df, lat="latitude", lon="longitude", z=w,
                radius=12, center={"lat": 40.73, "lon": -73.93},
                zoom=9, height=440, map_style="open-street-map"
            ),
            use_container_width=True,
        )

# 7) HOSTS & REVIEWS — distributions & barcharts
elif tab == "Hosts & Reviews":
    st.header("👤 Hosts & Reviews")

    st.subheader("1. Number of Reviews Distribution")
    st.plotly_chart(
        px.histogram(
            df_filtered,
            x="number_of_reviews",
            nbins=50,
            color="neighbourhood_group",
            title="Number of Reviews Distribution by Borough",
            log_y=True,
        ),
        use_container_width=True,
    )

    st.subheader("2. Reviews per Month Distribution")
    st.plotly_chart(
        px.histogram(
            df_filtered,
            x="reviews_per_month",
            nbins=50,
            color="room_type",
            title="Reviews per Month Distribution by Room Type",
            log_y=True,
        ),
        use_container_width=True,
    )

    st.subheader("3. Host Listings Count Distribution")
    st.plotly_chart(
        px.histogram(
            df_filtered,
            x="calculated_host_listings_count",
            nbins=30,
            color="neighbourhood_group",
            title="Host Listings Count Distribution by Borough",
            log_y=True,
        ),
        use_container_width=True,
    )

    st.subheader("4. Average Reviews per Month by Borough")
    avg_reviews_borough = (
        df_filtered.groupby("neighbourhood_group", as_index=False)["reviews_per_month"]
        .mean()
        .fillna(0)
        .sort_values(by="reviews_per_month", ascending=False)
    )
    fig = px.bar(
        avg_reviews_borough,
        x="neighbourhood_group",
        y="reviews_per_month",
        color="neighbourhood_group",
        text="reviews_per_month",
        title="Average Reviews per Month by Borough",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("5. Average Host Listings Count by Room Type")
    avg_host_listings_room = (
        df_filtered.groupby("room_type", as_index=False)["calculated_host_listings_count"]
        .mean()
        .sort_values(by="calculated_host_listings_count", ascending=False)
    )
    fig = px.bar(
        avg_host_listings_room,
        x="room_type",
        y="calculated_host_listings_count",
        color="room_type",
        text="calculated_host_listings_count",
        title="Average Host Listings Count by Room Type",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("6. Top 20 Hosts by Number of Listings (proxy)")
    top_hosts = (
        df_filtered.groupby("host_id")
        .agg(total_listings=("id", "count"), avg_price=("price", "mean"))
        .reset_index()
        .sort_values("total_listings", ascending=False)
        .head(20)
    )
    top_hosts["host_label"] = top_hosts["host_id"].astype(str)
    fig = px.bar(
        top_hosts,
        x="host_label",
        y="total_listings",
        color="host_label",
        title="Top 20 Hosts by Listings",
        hover_data=["avg_price"],
    )
    fig.update_layout(showlegend=False, xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("7. Reviews per Month vs Price (Scatter)")
    st.plotly_chart(
        px.scatter(
            df_filtered.dropna(subset=["reviews_per_month", "price"]),
            x="reviews_per_month",
            y="price",
            color="neighbourhood_group",
            hover_data=["room_type", "neighbourhood"],
            title="Price vs. Reviews per Month",
        ),
        use_container_width=True,
    )

    st.subheader("8. Reviews per Month vs Availability (Scatter)")
    st.plotly_chart(
        px.scatter(
            df_filtered.dropna(subset=["reviews_per_month", "availability_365"]),
            x="reviews_per_month",
            y="availability_365",
            color="room_type",
            hover_data=["neighbourhood_group"],
            title="Availability vs. Reviews per Month",
        ),
        use_container_width=True,
    )

    st.subheader("9. Room Type Share (Pie)")
    room_share = df_filtered["room_type"].value_counts().reset_index()
    room_share.columns = ["room_type", "count"]
    st.plotly_chart(
        px.pie(room_share, names="room_type", values="count", title="Room Type Share"),
        use_container_width=True,
    )

# 8) PREDICTION — Borough-picked 

elif tab == "Prediction":
    st.header("🔮 Price Prediction with XGBoost")
    st.markdown(
        "This quick predictor uses numeric inputs plus the selected **Borough** "
        "to infer location-related features from dataset averages."
    )

    # Pick borough; numerical inputs on the right
    pred_borough = st.selectbox("Choose Borough (NYC)", options=boroughs_all, index=0)

    col1, col2 = st.columns(2)
    with col1:
        minimum_nights = st.number_input("Minimum Nights", 1, 365, 3)
        number_of_reviews = st.number_input("Number of Reviews", 0, 10000, 25)
    with col2:
        reviews_per_month_default = float(np.round(df["reviews_per_month"].fillna(0).mean(), 2))
        reviews_per_month = st.number_input("Reviews per Month", 0.0, 50.0, reviews_per_month_default)
        availability_365 = st.slider("Availability (days)", 0, 365, 120)

    # Use borough means for location/host features so users don't have to input them
    borough_means = (
        df.groupby("neighbourhood_group")[["latitude", "longitude", "calculated_host_listings_count"]]
        .mean(numeric_only=True)
        .dropna()
    )
    if pred_borough not in borough_means.index:
        st.error("Selected borough has no statistics available in the dataset.")
        st.stop()

    b_lat = float(borough_means.loc[pred_borough, "latitude"])
    b_lon = float(borough_means.loc[pred_borough, "longitude"])
    b_host = float(borough_means.loc[pred_borough, "calculated_host_listings_count"])

    # Predict
    if st.button("Predict Price"):
        try:
            xgb_model = joblib.load("xgboost_airbnb.pkl")
        except FileNotFoundError:
            st.error("Model file 'xgboost_airbnb.pkl' not found. Please place it in the app directory.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # IMPORTANT: keep feature names/order exactly as in training
        input_data = pd.DataFrame(
            {
                "latitude": [b_lat],
                "longitude": [b_lon],
                "minimum_nights": [minimum_nights],
                "number_of_reviews": [number_of_reviews],
                "reviews_per_month": [reviews_per_month],
                "calculated_host_listings_count": [b_host],
                "availability_365": [availability_365],
            }
        )

        try:
            pred = float(xgb_model.predict(input_data)[0])
            pred = max(0.0, pred)  # no negative prices
            st.success(f"💰 Predicted Nightly Price in {pred_borough}: ${pred:.2f}")
        except Exception as e:
            st.error("Prediction error. Ensure training used the same feature names and order.")
            st.exception(e)
