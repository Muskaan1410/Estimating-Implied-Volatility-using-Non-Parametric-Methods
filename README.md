# Estimating-Implied-Volatility-using-Non-Parametric-Methods

#

## Overview

The accurate modeling of the implied volatility surface is crucial for option pricing, trading, and hedging. Traditional mathematical and machine-learning approaches often produce point estimates that can be misleading due to imperfections in market data. To address this, we employ Gaussian Processes (GPs), a Bayesian non-parametric method that generates posterior distributions of implied volatility functions. This allows us to quantify uncertainty and provide confidence intervals, offering a more robust approach to volatility estimation.

## Data

We train our model using historical SPX option data from 2020 to 2022, which includes end-of-day option quotes. This dataset provides a comprehensive representation of market conditions over three years, enhancing the robustness of our model.

## Training

1. **Preprocess Data:** Load and clean SPX option data using Postgres.
2. **Define Gaussian Process Model:** Use a kernel function to model the volatility surface.
3. **Optimize Hyperparameters:** Train using Variational Inference to scale for large datasets.
4. **Build Volatility Surface:** Construct a smooth implied volatility surface from model predictions.
5. **Evaluate Performance:** Compare against parametric models and assess predictive uncertainty.

## Streamlit App

An interactive **Streamlit** application has been developed to estmate implied volatility . To launch the app, run:

```bash
streamlit run app.py
```

This application allows users to explore different market conditions and analyze model predictions interactively.

##


