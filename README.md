# Credit-Risk-Modelling

## Live Project
View the complete project analysis and implementation at [GitHub Pages](https://majidjangani.github.io/Credit-Risk-Modelling/Banks-Pr(default)/)

**Corporate Default** occurs when a company fails to meet its debt obligations, such as missing payments or filing for bankruptcy. Predicting bank defaults is a closely related challenge, but it comes with unique complexities:

- **Key Differences Between Banks and Corporations:**
  - Banks do not sell physical or digital goods; their financial statements (Call Reports) lack metrics like the cost of goods sold.
  - Banks have a more intricate debt structure, focusing on lending and deposits.
  - Their assets are highly specialized compared to corporations' non-financial assets.

By examining financial ratios instead of raw data, we can effectively identify patterns that indicate a binary outcome: **default** or **non-default**.

## Data Source
The project utilizes data from the **Federal Deposit Insurance Corporation (FDIC)** to model the probability of default for banks. This data is analyzed to derive key financial ratios and economic features critical for prediction.


 **Machine Learning Techniques:**
   - **Classification and Regression Trees (CART):** Initial model for predicting default probabilities.
   - **Random Forest Algorithm:** Identifies the most influential features within the tree-based model.
   - **Extreme Gradient Boosting (XGBoost):** Optimizes the model for enhanced predictive accuracy.

## Repository Structure
```python
Creadit-Risk-Modelling-/
root/
/
├── _includes/
│   └── head-custom.html
├── assets/
│   └── style.scss
├── Banks-pr(default).md 
├── README.md
├── _config.yml
└── index.md 
