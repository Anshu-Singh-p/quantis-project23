import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Quantis Smart Store",
    layout="wide",
    page_icon="🛍️"
)

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
    <style>
        .main {
            background-color: #0f1117;
        }
        .product-card {
            background: #1c1f26;
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
            text-align: center;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #ffffff;
        }
        .subtitle {
            color: #a0a0a0;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
required_features = list(model.feature_names_in_)

# ---------------- SESSION STATE ----------------
for key in ["views", "cart", "time_spent", "cart_items"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key != "cart_items" else []

# ---------------- PRODUCTS ----------------
products = [
    {"name": "Smartphone", "price": 15000, "img": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9", "rating": 4.5, "category": "Electronics"},
    {"name": "Laptop", "price": 55000, "img": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8", "rating": 4.2, "category": "Electronics"},
    {"name": "Smartwatch", "price": 3000, "img": "https://images.unsplash.com/photo-1516574187841-cb9cc2ca948b", "rating": 4.1, "category": "Electronics"},
    {"name": "Tablet", "price": 20000, "img": "https://images.unsplash.com/photo-1542751371-adc38448a05e", "rating": 4.3, "category": "Electronics"},
    {"name": "Shoes", "price": 2500, "img": "https://images.unsplash.com/photo-1528701800489-20be3c6d9d4d", "rating": 4.0, "category": "Fashion"},
    {"name": "Backpack", "price": 1200, "img": "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c", "rating": 3.9, "category": "Fashion"},
    {"name": "Gaming Mouse", "price": 900, "img": "https://images.unsplash.com/photo-1587202372775-e229f172b9d7", "rating": 4.3, "category": "Electronics"},
    {"name": "Keyboard", "price": 1500, "img": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8", "rating": 4.1, "category": "Electronics"},
    {"name": "Monitor", "price": 12000, "img": "https://images.unsplash.com/photo-1587825140708-dfaf72ae4b04", "rating": 4.4, "category": "Electronics"},
    {"name": "Power Bank", "price": 1000, "img": "https://images.unsplash.com/photo-1580894894513-541e068a3e2b", "rating": 4.2, "category": "Electronics"},
]

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Smart Controls")
page = st.sidebar.radio("Navigate", ["Home", "Cart", "Analytics"])
search = st.sidebar.text_input("Search Product")
category = st.sidebar.selectbox("Category", ["All", "Electronics", "Fashion"])

# ---------------- FILTER ----------------
filtered_products = [p for p in products if search.lower() in p["name"].lower() and (category == "All" or p["category"] == category)]

# ---------------- HOME ----------------
if page == "Home":
    st.markdown("<div class='title'>🛍️ Quantis Smart Store</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Smart shopping powered by AI</div>", unsafe_allow_html=True)

    cols = st.columns(4)

    for i, product in enumerate(filtered_products):
        with cols[i % 4]:
            st.image(product["img"], use_column_width=True)
            st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
            st.subheader(product["name"])
            st.write(f"💰 ₹{product['price']}")
            st.write(f"⭐ {product['rating']}")

            if st.button("View", key=f"view{i}"):
                st.session_state.views += 1
                st.session_state.time_spent += np.random.randint(5, 15)

            if st.button("Add to Cart", key=f"cart{i}"):
                st.session_state.cart += 1
                st.session_state.cart_items.append(product)

            st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CART ----------------
elif page == "Cart":
    st.title("🛒 Cart Summary")

    total = 0
    for item in st.session_state.cart_items:
        st.write(f"{item['name']} - ₹{item['price']}")
        total += item["price"]

    st.metric("Total Spend", f"₹{total}")

# ---------------- ANALYTICS ----------------
elif page == "Analytics":
    st.title("📊 User Behavior Analytics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Views", st.session_state.views)
    col2.metric("Cart Adds", st.session_state.cart)
    col3.metric("Time Spent", st.session_state.time_spent)

    fig, ax = plt.subplots()
    ax.bar(["Views", "Cart", "Time"], [st.session_state.views, st.session_state.cart, st.session_state.time_spent])
    st.pyplot(fig)

# ---------------- ML PREDICTION ----------------
st.sidebar.subheader("📊 AI Prediction")

views = st.session_state.views
cart = st.session_state.cart
time_spent = st.session_state.time_spent

data = {}
for f in required_features:
    if f == "avg_spent": data[f] = 3000 + cart * 1500
    elif f == "cart_additions": data[f] = cart
    elif f == "discount": data[f] = min(0.1 + views * 0.01, 0.5)
    elif f == "past_purchases": data[f] = max(1, int(time_spent / 10))
    elif f == "price": data[f] = 1000 + views * 200
    elif f == "rating": data[f] = 4.0 + cart * 0.1
    elif f == "time_spent": data[f] = time_spent
    elif f == "view_count": data[f] = views
    else: data[f] = 0

input_data = pd.DataFrame([data])
base_prob = model.predict_proba(input_data)[0][1]
engagement = min((views*0.05 + cart*0.25 + time_spent*0.01), 1)
final_prob = min(0.7 * base_prob + 0.3 * engagement, 0.95)

st.sidebar.success(f"Purchase Probability: {final_prob:.2f}")

