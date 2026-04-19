# generate_data.py

import pandas as pd
import numpy as np

n = 5000

data = {
    "view_count": np.random.randint(1, 10, n),
    "cart_additions": np.random.randint(0, 5, n),
    "time_spent": np.random.randint(10, 300, n),
    "price": np.random.randint(100, 5000, n),
    "discount": np.random.randint(0, 70, n),
    "rating": np.random.uniform(1, 5, n),
    "past_purchases": np.random.randint(0, 50, n),
    "avg_spent": np.random.randint(100, 5000, n),
}

df = pd.DataFrame(data)

# Create realistic target
df["purchased"] = (
    (df["cart_additions"] > 1) |
    (df["time_spent"] > 120) |
    (df["discount"] > 30)
).astype(int)

df.to_csv("Ecommerce_fixed.csv", index=False)