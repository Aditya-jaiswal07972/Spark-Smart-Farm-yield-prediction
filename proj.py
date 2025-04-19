import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder.appName("CropYieldDashboard").getOrCreate()

data = spark.read.csv("smart_farming_crop_yield_prediction.csv", header=True, inferSchema=True)

features = data.columns[:-1]
assembler = VectorAssembler(inputCols=features, outputCol="features")
data = assembler.transform(data)

train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Train models
lr = LinearRegression(featuresCol="features", labelCol="Yield_kg_per_ha")
rf = RandomForestRegressor(featuresCol="features", labelCol="Yield_kg_per_ha", numTrees=100, seed=42)

lr_model = lr.fit(train_data)
rf_model = rf.fit(train_data)

# Predictions
lr_preds = lr_model.transform(test_data)
rf_preds = rf_model.transform(test_data)

# Evaluation
# RMSE Score 
evaluator = RegressionEvaluator(labelCol="Yield_kg_per_ha", predictionCol="prediction", metricName="rmse")
lr_rmse = evaluator.evaluate(lr_preds)
rf_rmse = evaluator.evaluate(rf_preds)
# R2 Score
r2_evaluator = RegressionEvaluator(labelCol="Yield_kg_per_ha", predictionCol="prediction", metricName="r2")
lr_r2 = r2_evaluator.evaluate(lr_preds)
rf_r2 = r2_evaluator.evaluate(rf_preds)


# Convert to Pandas
lr_df = lr_preds.select("Yield_kg_per_ha", "prediction").toPandas()
rf_df = rf_preds.select("Yield_kg_per_ha", "prediction").toPandas()
data_pd = data.select(data.columns).toPandas()

numeric_data = data_pd.select_dtypes(include=['number'])
corr_matrix = numeric_data.corr()


# Feature Importances
importances = rf_model.featureImportances
importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances.toArray()
}).sort_values("importance", ascending=False)

lr_df["residuals"] = lr_df["Yield_kg_per_ha"] - lr_df["prediction"]

def run_dashboard():
    # Layout Setup
    st.set_page_config(layout="wide")
    st.title("🌾 Smart Farming: Crop Yield Prediction Dashboard")
    st.markdown("An interactive dashboard showcasing model insights and yield data analysis.")

    # KPI Cards
    col1, col2 = st.columns(2)
    col1.metric("📈 Linear Regression RMSE", f"{lr_rmse:.2f}")
    col1.metric("✅ Linear Regression R²", f"{lr_r2:.2f}")

    col2.metric("🌳 Random Forest RMSE", f"{rf_rmse:.2f}")
    col2.metric("✅ Random Forest R²", f"{rf_r2:.2f}")
    st.markdown("Lower RMSE and Higher R² (max 1.0) Mean Better Model Performance")

    # Row 1: Linear Regression Results
    st.markdown("### 📌 Linear Regression Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual vs Predicted")
        fig = px.scatter(lr_df, x="Yield_kg_per_ha", y="prediction",
                         title="Actual vs Predicted (LR)",
                         labels={"Yield_kg_per_ha": "Actual", "prediction": "Predicted"})
        fig.add_shape(type='line', x0=lr_df.min().min(), x1=lr_df.max().max(),
                      y0=lr_df.min().min(), y1=lr_df.max().max(),
                      line=dict(dash='dash', color='red'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Residual Distribution")
        lr_df["residuals"] = lr_df["Yield_kg_per_ha"] - lr_df["prediction"]
        fig_res = px.histogram(lr_df, x="residuals", nbins=30, title="Residuals (LR)")
        st.plotly_chart(fig_res, use_container_width=True)

    # Row 2: Yield Distribution & Nitrogen Impact
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌾 Yield Distribution")
        fig_yield = px.histogram(
            data_pd,
            x="Yield_kg_per_ha",
            nbins=40,
            title="🌾 Yield per Hectare Distribution",
            color_discrete_sequence=['green'],
            labels={"Yield_kg_per_ha": "Yield (kg/ha)", "count": "Number of Fields"}
        )
        st.plotly_chart(fig_yield, use_container_width=True)

    with col2:
        st.subheader("🌿 Nitrogen vs Yield")
        fig_n = px.scatter(data_pd, x='Nitrogen_kg_ha', y='Yield_kg_per_ha',
                           color='Nitrogen_kg_ha',
                           title="Impact of Nitrogen on Crop Yield",
                           labels={'Nitrogen_kg_ha': 'Nitrogen (kg/ha)', 'Yield_kg_per_ha': 'Yield (kg/ha)'},
                           color_continuous_scale="YlGn")
        st.plotly_chart(fig_n, use_container_width=True)

    # Row 3: Feature Importance and Correlation
    st.markdown("### 🔍 Feature Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌱 Feature Importances (RF)")
        fig_feat = px.bar(importance_df, x="importance", y="feature", orientation='h',
                          title="Feature Importances from Random Forest",
                          color="importance", color_continuous_scale="Viridis")
        st.plotly_chart(fig_feat, use_container_width=True)

    with col2:
        st.subheader("🎯 Top 5 Influential Features (Pie)")
        top5 = importance_df.head(5)
        fig_pie = px.pie(top5, names='feature', values='importance', title='Top 5 Feature Contributions')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Correlation Heatmap
    label_mapping = {
        'Nitrogen_kg_ha': 'Nitro',
        'Phosphorus_kg_ha': 'P',
        'Potassium_kg_ha': 'K',
        'Temperature_C': 'Temp',
        'Humidity_percent': 'Humid',
        'Soil_pH': 'pH',
        'Rainfall_mm': 'Rain',
        'Yield_kg_per_ha': 'Yield'
    }
    corr_matrix_labeled = corr_matrix.rename(index=label_mapping, columns=label_mapping)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Correlation Heatmap")
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix_labeled, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5, ax=ax)
        plt.xticks(rotation=0, ha='center')
        plt.yticks(rotation=0, va='center')
        st.pyplot(fig_corr)

    # Footer
    st.markdown("---")
    st.markdown("Made with using Streamlit, PySpark, and Plotly.           — **by Aditya**")

if __name__ == "__main__":
    run_dashboard()