import streamlit as st
import torch
import gpytorch
import numpy as np
import pandas as pd
import pickle
import datetime
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define Preprocessing Function
def preprocess_data(df, scaler):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Standardize features
    numeric_cols = ["Moneyness", "time_to_maturity_days"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df

# Define the SVGP Model
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inducing_points = torch.randn(500, 2, dtype=torch.float64).to(device)  # Dummy inducing points
model = SVGPModel(inducing_points).to(device).double()
model.load_state_dict(torch.load("gp_model.pth", map_location=device))
model.eval()

# Black-Scholes Function
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price

# Streamlit UI
st.set_page_config(page_title="SPX Options Pricing", layout="wide")

# Custom CSS for Centered Title
st.markdown(
    """
    <style>
    .stTitle {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>📊 SPX Options Implied Volatility & Black-Scholes Pricing</h1>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("🔢 Input Parameters")

stock_price = st.sidebar.number_input("Stock Price (S)", min_value=1.0, value=100.0)
strike_price = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=100.0)
maturity_date = st.sidebar.date_input("Option Maturity Date", min_value=datetime.date.today() + datetime.timedelta(days=1))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

# Compute Moneyness & Time to Maturity
moneyness = stock_price / strike_price
time_to_maturity_days = (maturity_date - datetime.date.today()).days

# Predict Button
if st.sidebar.button("🚀 Predict IV & Option Prices"):
    with st.spinner("Crunching numbers... ⏳"):
        time.sleep(1)  # Simulate processing time

        # Prepare Data
        df_input = pd.DataFrame({"Moneyness": [moneyness], "time_to_maturity_days": [time_to_maturity_days]})
        df_processed = preprocess_data(df_input, scaler)
        X_input_torch = torch.tensor(df_processed.values, dtype=torch.float64).to(device)

        # Predict IV
        with torch.no_grad():
            pred = model(X_input_torch)
            iv_pred = np.exp(pred.mean.cpu().numpy()[0])  # Convert from log scale

        # Calculate Option Prices
        call_price, put_price = black_scholes(stock_price, strike_price, time_to_maturity_days / 365, risk_free_rate, iv_pred)

        # Display Results
        st.success(f"✅ Predicted Implied Volatility: **{iv_pred:.4f}**")
        st.info(f"📈 Call Option Price: **${call_price:.2f}**")
        st.info(f"📉 Put Option Price: **${put_price:.2f}**")

        # Generate Dynamic Line Chart
        time_range = np.linspace(1, time_to_maturity_days, 50)
        call_prices = []
        put_prices = []

        for t in time_range:
            c_price, p_price = black_scholes(stock_price, strike_price, t / 365, risk_free_rate, iv_pred)
            call_prices.append(c_price)
            put_prices.append(p_price)

        # Plot Option Prices vs. Time to Maturity
        fig, ax = plt.subplots()

        # Plot Call and Put Option Prices
        ax.plot(time_range, call_prices, label="Call Option Price", color="green", linewidth=2)
        ax.plot(time_range, put_prices, label="Put Option Price", color="red", linewidth=2)

        # Add a vertical dotted line at the maturity date
        ax.axvline(time_to_maturity_days, color="blue", linestyle="dotted", linewidth=2, label="Maturity Date")

        # Mark the maturity date on x-axis
        ax.set_xticks(list(ax.get_xticks()) + [time_to_maturity_days])
        ax.set_xticklabels([str(int(tick)) for tick in ax.get_xticks()])
        ax.set_xlabel("Time to Maturity (Days)")

        # Labels and Title
        ax.set_ylabel("Option Price ($)")
        ax.set_title("📉 Option Prices vs. Time to Maturity")
        ax.legend()
        ax.grid(True)

        # Display the updated plot
        st.pyplot(fig)
