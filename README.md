# Customer Segmentation & Service Recommendation System

## 📌 Project Overview

This project develops a customer analytics system that segments
customers based on behavioral, engagement, spending and learning
interest patterns.

The system uses K-Means clustering to identify customer groups
and applies rule-based recommendation logic to recommend suitable
services and learning programs.

---

## 🎯 Objectives

- Perform customer behavior analysis
- Engineer meaningful customer features
- Apply K-Means clustering
- Evaluate cluster quality
- Create meaningful customer segments
- Recommend suitable services
- Build an interactive Streamlit dashboard

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- K-Means
- PCA
- Streamlit
- Jupyter Notebook

---

## 🔄 Project Workflow

Customer Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Standardization
        ↓
K-Means Clustering
        ↓
Elbow + Silhouette Evaluation
        ↓
Customer Segmentation
        ↓
Recommendation Logic
        ↓
Streamlit Dashboard

---

## 📊 Machine Learning Approach

K-Means clustering was selected because the project focuses on
unsupervised customer segmentation.

Numerical features were standardized using StandardScaler.

The optimal number of clusters was evaluated using:

1. Elbow Method
2. Silhouette Score

PCA was then used to visualize the resulting clusters in two dimensions.

---

## 💡 Recommendation System

The recommendation engine uses business rules based on:

- Customer engagement
- Monthly spending
- Learning interest
- Service inquiries

This approach provides transparent and interpretable recommendations.
