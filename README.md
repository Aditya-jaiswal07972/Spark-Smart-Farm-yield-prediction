# 🌾 Smart Farming Crop Yield Prediction Dashboard

An interactive dashboard for analyzing and predicting crop yield using agricultural input data. Built with **PySpark**, **Streamlit**, **Plotly**, **Pandas**, and packaged with **Unified Python Packaging (uv)**.

---

## 🚀 Features

- 📈 Predict crop yield using **Linear Regression** and **Random Forest**
- 🧮 Visualize **actual vs. predicted yield** and **residuals**
- 📊 Interactive charts, correlation heatmaps, and feature importance
- 🧩 Dynamic filters and dropdowns for granular data exploration
- 🧼 Clean, responsive layout with a smooth UX using Streamlit

---

## 📷 Screenshots

| Actual vs Predicted |
|---------------------|
![Actual vs Predicted](https://github.com/Aditya-jaiswal07972/Spark-Smart-Farm-yield-prediction/blob/main/assets/acutal_VS_prediction.png?raw=true)

| Nitrogen vs Yield |
|-------------------|
![Nitrogen vs Yield](https://github.com/Aditya-jaiswal07972/Spark-Smart-Farm-yield-prediction/blob/06471835e1fd6fe408cee2daaae87c53ec48b176/assets/NitroVSyeild.png)

---

## 🗂️ Project Structure

```
smart-farm-predic/
│
├── proj.py                  # Main Streamlit dashboard logic
├── main.py                  # Optional entry point (can run proj.py)
├── smart_farming_crop_yield_prediction.csv
├── pyproject.toml           # Project metadata and dependencies (uv)
├── uv.lock                  # Locked dependencies (auto-generated)
├── .venv/                   # Virtual environment (optional)
├── assets/                  # Images and visual assets
└── README.md
```

---

## 🛠️ Getting Started

Make sure Python `>=3.10` and [`uv`](https://github.com/astral-sh/uv) are installed.

### 1. Clone the Repository

```bash
git clone https://github.com/Aditya-jaiswal07972/Spark-Smart-Farm-yield-prediction.git
cd smart-farm-predic
```

### 2. Install Dependencies with `uv`

Using a virtual environment:

```bash
uv venv
uv pip install
```

Or install globally:

```bash
uv pip install --system
```

### 3. Run the App

```bash
# Option 1: Directly run the Streamlit app
streamlit run proj.py

# Option 2: Use the CLI entry point
python main.py
```

---

## 📦 Tech Stack

All dependencies are managed via `pyproject.toml`. Key packages:

- `streamlit` – for building the interactive dashboard
- `pyspark` – for data processing at scale
- `pandas`, `plotly-express`, `seaborn`, `matplotlib` – for analysis and visualization

---

## 👨‍🌾 Author

Made with ❤️ by [**Aditya Jaiswal**](https://github.com/Aditya-jaiswal07972)

## 📄 License

Released under the [MIT License](https://opensource.org/licenses/MIT). Feel free to fork, use, and contribute!
