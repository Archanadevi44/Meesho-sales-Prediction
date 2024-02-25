# Meesho-sales-Prediction
Meesho Sales Prediction is a forecasting tool that uses data analysis to predict future sales on the Meesho platform, helping sellers make informed decisions about inventory, pricing, and marketing strategies.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv("/content/MEESHO1.csv")
data.head()

data.drop(["Supplier Discounted Price (Incl GST and Commision)"],inplace=True,axis=1)
data.columns = ['Order Date', 'sub_order_num', 'order_status', 'state', 'pin', 'gst_amount', 'meesho_price', 'shipping_charges_total', 'price', 'delivered_date', 'Product Name', 'SKU', 'Size', 'Quantity', 'Supplier Listed Price (Incl. GST + Commission)']
data.info()
plt.figure(figsize=(10,7))
plt.subplot(1,2,1)
plt.pie(data.groupby("order_status")["price"].sum(),autopct="%0.2f%%",labels=data["order_status"].unique())
plt.show()

plt.figure(figsize=(15,4))
sns.barplot(data=data,x="Size",y="price")
sns.histplot(data["price"])
sns.barplot(y = data.groupby("order_status")["price"].sum(),x = data["order_status"].unique())
plt.title("Order status vs Price")
plt.show()

sns.barplot(x=data["Order Date"].unique() ,y = data.groupby("Order Date")["price"].sum())
plt.title("Date vs Oreder")
plt.xticks(rotation=70)
plt.show()

plt.figure(figsize=(15,5))
sns.barplot(x=data["state"].unique() ,y = data.groupby("state")["price"].sum())
plt.title("State vs Price")
plt.xticks(rotation=70)
plt.show()

deliverd = pd.DataFrame(data[data["order_status"]=="DELIVERED"])
deliverd
data["Order Date"] = pd.to_datetime(data["Order Date"])
deliverd["Order Date"] = pd.to_datetime(deliverd["Order Date"])
data["day_name"] = data["Order Date"].dt.day_name()
data["day"] = data["Order Date"].dt.day
deliverd["day_name"] = deliverd["Order Date"].dt.day_name()
deliverd["day"] = deliverd["Order Date"].dt.day
plt.figure(figsize=(15,5))
sns.barplot(x=data["day_name"].unique() ,y = data.groupby("day_name")["price"].sum())
plt.title("Day vs Order")
plt.xticks(rotation=70)
plt.show()

plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(data["Order Date"].unique(), data.groupby("Order Date")["meesho_price"].sum())
plt.title("Order Date vs Actual Price")
plt.subplot(1,2,2)
plt.plot(data["Order Date"].unique(), data.groupby("Order Date")["meesho_price"].count())
plt.title("Order Date vs Order Count")
pred_df = data.groupby("Order Date")["price"].count().to_frame()
pred_df.reset_index(inplace=True)
pred_df.columns = ["ds","y"]
pred_df.head(3)

!pip install cmdstanpy==0.9.5
!pip install pystan==2.19.1.1 --no-cache-dir
!pip install prophet
# Import prophet
from prophet import Prophet
from prophet import Prophet
m = Prophet()
m.fit(pred_df)  # df is a pandas.DataFrame with 'y' and 'ds' columns
future = m.make_future_dataframe(periods=10)
predict = m.predict(future)

fig1 = m.plot(predict)

plt.figure(figsize=(15,5))
plt.plot(predict["ds"],predict["yhat"],label="Prediced")
plt.plot(pred_df["ds"],pred_df["y"],label="Actual")
plt.title("10 Day order Prediction")
plt.legend()
