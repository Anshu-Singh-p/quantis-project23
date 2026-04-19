import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quantis Smart Store", layout="wide")

# -------- LOAD MODEL --------
model = joblib.load("model.pkl")

# 🔥 GET REQUIRED FEATURES AUTOMATICALLY
required_features = list(model.feature_names_in_)

# -------- SESSION STATE --------
if "views" not in st.session_state:
    st.session_state.views = 0
if "cart" not in st.session_state:
    st.session_state.cart = 0
if "time_spent" not in st.session_state:
    st.session_state.time_spent = 0
if "cart_items" not in st.session_state:
    st.session_state.cart_items = []

# -------- PRODUCTS --------
products = [
    {"name": "Smartphone", "price": 15000, "img": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9", "rating": 4.5, "category": "Electronics"},
    {"name": "Laptop", "price": 55000, "img": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8", "rating": 4.2, "category": "Electronics"},
    {"name": "Headphones", "price": 2000, "img": "https://images.unsplash.com/photo-1518441902110-72e6b5b4f7c6", "rating": 4.0, "category": "Electronics"},
    {"name": "Smartwatch", "price": 3000, "img": "https://images.unsplash.com/photo-1516574187841-cb9cc2ca948b", "rating": 4.1, "category": "Electronics"},
    {"name": "Tablet", "price": 20000, "img": "https://images.unsplash.com/photo-1542751371-adc38448a05e", "rating": 4.3, "category": "Electronics"},
    {"name": "Camera", "price": 40000, "img": "https://images.unsplash.com/photo-1519183071298-a2962be96c54", "rating": 4.4, "category": "Electronics"},
    {"name": "Shoes", "price": 2500, "img": "https://images.unsplash.com/photo-1528701800489-20be3c6d9d4d", "rating": 4.0, "category": "Fashion"},
    {"name": "Backpack", "price": 1200, "img": "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c", "rating": 3.9, "category": "Fashion"},
    {"name": "Bluetooth Speaker", "price": 1800, "img": "https://images.unsplash.com/photo-1585386959984-a41552231658", "rating": 4.2, "category": "Electronics"},
    {"name": "Gaming Mouse", "price": 900, "img": "https://images.unsplash.com/photo-1587202372775-e229f172b9d7", "rating": 4.3, "category": "Electronics"},
    {"name": "Keyboard", "price": 1500, "img": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8", "rating": 4.1, "category": "Electronics"},
    {"name": "Monitor", "price": 12000, "img": "https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04", "rating": 4.4, "category": "Electronics"},
    {"name": "Power Bank", "price": 1000, "img": "https://images.unsplash.com/photo-1580894894513-541e068a3e2b", "rating": 4.2, "category": "Electronics"},
    {"name": "Printer", "price": 8000, "img": "https://images.unsplash.com/photo-1581091012184-5c1c6c52b4c6", "rating": 4.0, "category": "Electronics"},
    {"name": "Earbuds", "price": 2200, "img": "https://images.unsplash.com/photo-1580894908361-967195033215", "rating": 4.3, "category": "Electronics"},
]

# -------- SIDEBAR --------
st.sidebar.title("🛒 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Cart"])

search = st.text_input("🔍 Search Product")
category = st.selectbox("📂 Category", ["All", "Electronics", "Fashion"])

filtered_products = [
    p for p in products
    if search.lower() in p["name"].lower() and (category == "All" or p["category"] == category)
]

# -------- HOME --------
if page == "Home":
    st.title("🛍️ Quantis Smart Store")

    for i in range(0, len(filtered_products), 5):
        cols = st.columns(5)
        for j in range(5):
            if i + j < len(filtered_products):
                product = filtered_products[i + j]
                with cols[j]:
                    st.image(product["img"], use_column_width=True)
                    st.subheader(product["name"])
                    st.write(f"₹{product['price']}")
                    st.write(f"⭐ {product['rating']}")

                    if st.button("View", key=f"view{i+j}"):
                        st.session_state.views += 1
                        st.session_state.time_spent += np.random.randint(5, 15)

                    if st.button("Add to Cart", key=f"cart{i+j}"):
                        st.session_state.cart += 1
                        st.session_state.cart_items.append(product)

# -------- CART --------
if page == "Cart":
    st.title("🛒 Your Cart")

    if len(st.session_state.cart_items) == 0:
        st.write("Cart is empty")
    else:
        total = 0
        for item in st.session_state.cart_items:
            st.write(f"{item['name']} - ₹{item['price']}")
            total += item["price"]

        st.subheader(f"Total: ₹{total}")

# -------- ML PREDICTION (AUTO MATCH) --------
st.sidebar.subheader("📊 Prediction Engine")

views = st.session_state.views
cart = st.session_state.cart
time_spent = st.session_state.time_spent

# Create empty input with correct columns
data = {}

for feature in required_features:
    if feature == "avg_spent":
        data[feature] = 3000 + (cart * 1500)
    elif feature == "cart_additions":
        data[feature] = cart
    elif feature == "discount":
        data[feature] = min(0.1 + views * 0.01, 0.5)
    elif feature == "past_purchases":
        data[feature] = max(1, int(time_spent / 10))
    elif feature == "price":
        data[feature] = 1000 + (views * 200)
    elif feature == "rating":
        data[feature] = 4.0 + (cart * 0.1)
    elif feature == "time_spent":
        data[feature] = time_spent
    elif feature == "view_count":
        data[feature] = views
    else:
        data[feature] = 0  # fallback

input_data = pd.DataFrame([data])

# Predict
base_prob = model.predict_proba(input_data)[0][1]

engagement_score = min((views*0.05 + cart*0.25 + time_spent*0.01), 1)
final_prob = min(0.7 * base_prob + 0.3 * engagement_score, 0.95)

# Display
st.sidebar.write(f"Views: {views}")
st.sidebar.write(f"Cart: {cart}")
st.sidebar.write(f"Time: {time_spent}")
st.sidebar.success(f"Purchase Probability: {final_prob:.2f}")

# -------- GRAPH --------
st.sidebar.subheader("📈 Behavior Graph")

fig, ax = plt.subplots()
ax.bar(["Views", "Cart", "Time"], [views, cart, time_spent])
st.sidebar.pyplot(fig)