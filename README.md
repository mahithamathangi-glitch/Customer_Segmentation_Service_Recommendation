# 📊 Customer Segmentation & Service Recommendation System

## 📌 Project Overview

The **Customer Segmentation & Service Recommendation System** is an end-to-end Data Science project designed to analyze customer behavior, identify meaningful customer segments, and recommend suitable services based on customer characteristics.

The project uses **K-Means clustering**, an unsupervised machine learning algorithm, to group customers according to behavioral, engagement, spending, service interaction, and learning-interest patterns.

A transparent **rule-based recommendation engine** is then used to recommend suitable services for individual customers.

An interactive **Streamlit dashboard** provides customer insights, segment visualization, customer profiles, and service recommendations.

---

## 🎯 Project Objectives

The major objectives of this project are:

- Analyze customer behavioral patterns
- Perform data preprocessing and exploratory data analysis
- Engineer meaningful customer-level features
- Standardize numerical features
- Apply K-Means clustering for customer segmentation
- Determine an appropriate number of clusters
- Evaluate clustering quality using the Elbow Method and Silhouette Score
- Create meaningful business-oriented customer segments
- Visualize customer segments using PCA
- Develop a transparent service recommendation system
- Build an interactive Streamlit dashboard
- Generate actionable business recommendations
- Maintain a version-controlled and rollback-ready project using Git

---

## 💼 Business Problem

Organizations often have customers with different:

- Spending patterns
- Engagement levels
- Purchase frequencies
- Learning interests
- Service requirements
- Platform usage behavior

Treating every customer in the same way can result in ineffective marketing, poor personalization, and inefficient service delivery.

This project addresses the problem by identifying groups of customers with similar characteristics and providing targeted recommendations for each group.

---

# 🔄 Project Workflow

```text
                 CUSTOMER DATA
                       │
                       ▼
              DATA PREPROCESSING
                       │
                       ▼
              EXPLORATORY DATA ANALYSIS
                       │
                       ▼
              FEATURE ENGINEERING
                       │
                       ▼
                STANDARDIZATION
                       │
                       ▼
                 K-MEANS MODEL
                       │
              ┌────────┴────────┐
              ▼                 ▼
        ELBOW METHOD      SILHOUETTE SCORE
              │                 │
              └────────┬────────┘
                       ▼
              CUSTOMER SEGMENTS
                       │
              ┌────────┴────────┐
              ▼                 ▼
             PCA          SEGMENT PROFILES
              │                 │
              └────────┬────────┘
                       ▼
             RECOMMENDATION ENGINE
                       │
                       ▼
              STREAMLIT DASHBOARD
                       │
                       ▼
               BUSINESS INSIGHTS

📂 Dataset
The project uses synthetic customer data created for demonstration and educational purposes.
The dataset contains customer-level behavioral and engagement information.
Main Features
Feature	Description
Customer_ID	Unique customer identifier
Age	Customer age
Tenure_Months	Duration of customer relationship
Monthly_Spend	Monthly customer spending
Purchase_Frequency	Number of purchases
Engagement_Score	Customer engagement level
Service_Inquiries	Number of service-related inquiries
Learning_Interest	Interest in learning programs
Avg_Session_Minutes	Average platform session duration


🧹 Data Preprocessing
The following preprocessing steps were performed:
1. Dataset loading
2. Data structure inspection
3. Missing-value verification
4. Statistical summary
5. Numerical feature analysis
6. Feature engineering
7. Feature scaling
Numerical features were standardized using:
StandardScaler()


This was necessary because the features have different numerical ranges.
For example, monthly spending can be thousands of units while engagement scores may range from 0–100.
Standardization prevents large-scale variables from dominating the distance calculations used by K-Means.
🛠️ Feature Engineering
Additional behavioral features were created to improve customer analysis.
1. Spend per Purchase
Spend_per_Purchase =
Monthly_Spend / Purchase_Frequency

This estimates the average spending associated with each purchase.
2. Engagement per Session
Engagement_per_Session =
Engagement_Score / Avg_Session_Minutes

This provides an additional measure of engagement intensity.
3. Customer Value Score
A composite score was created using:
- Monthly spending
- Purchase frequency
- Engagement score
This provides an interpretable estimate of customer value.
🤖 Machine Learning Methodology
K-Means Clustering
K-Means was selected because the objective of the project is to identify natural groups of customers without predefined target labels.
The algorithm works by:
1. Selecting the number of clusters
2. Initializing cluster centroids
3. Assigning customers to the nearest centroid
4. Updating the centroids
5. Repeating until the clusters stabilize
📐 Cluster Evaluation
Two techniques were used to evaluate the clustering structure.
1. Elbow Method
The Elbow Method evaluates the relationship between:
- Number of clusters
- Within-cluster sum of squares / inertia
The point where the reduction in inertia begins to slow down provides an indication of a suitable cluster count.
The resulting visualization is available in:
images/elbow_curve.png

2. Silhouette Score
The Silhouette Score measures how well each customer fits within its assigned cluster compared with other clusters.
A higher score generally indicates better-separated and more cohesive clusters.
The evaluation is available in:
images/silhouette_score.png

👥 Customer Segmentation
The final customer groups are given business-friendly names based on their dominant behavioral characteristics.
Premium Engaged Customers
Customers showing comparatively strong customer value, spending and engagement characteristics.
Recommended strategy:
- Premium memberships
- Advanced services
- Loyalty benefits
- Personalized offers
Learning-Focused Customers
Customers showing strong interest in learning and educational opportunities.
Recommended strategy:
- Advanced learning programs
- Data Science / AI programs
- Certifications
- Skill-development programs
Service-Support Customers
Customers showing comparatively high service interaction or support requirements.
Recommended strategy:
- Priority customer support
- Proactive service assistance
- Faster issue resolution
- Dedicated support channels
Occasional Users
Customers with comparatively lower engagement or spending activity.
Recommended strategy:
- Engagement campaigns
- Starter programs
- Personalized offers
- Re-engagement campaigns
Segment names are assigned based on the behavioral characteristics of the generated clusters rather than being predefined customer labels.

📊 PCA Visualization
Since the clustering model uses multiple dimensions, Principal Component Analysis (PCA) was used to project the customer data into two dimensions.
This allows the resulting customer segments to be visually inspected.
The visualization is available at:
images/cluster_visualization.png

🎯 Service Recommendation System
The recommendation layer uses transparent rule-based logic.
Recommendations are generated using customer characteristics such as:
- Learning interest
- Monthly spending
- Engagement
- Service inquiries
Example recommendation logic:
High Learning Interest
        ↓
Advanced Learning Program

High Spending + High Engagement
        ↓
Premium Membership

High Service Inquiries
        ↓
Priority Customer Support

Low Engagement
        ↓
Engagement Booster

Other Customers
        ↓
Starter Learning & Service Package

The rule-based approach was intentionally selected because it is:
- Easy to understand
- Transparent
- Explainable
- Easy to modify
- Suitable for business deployment
🖥️ Interactive Dashboard
The project includes a Streamlit dashboard that allows users to explore the customer segments interactively.
Dashboard Features
- Total customer count
- Average customer spending
- Average engagement score
- Average learning interest
- Customer segment distribution
- Interactive segment filtering
- 2D PCA cluster visualization
- Segment profile analysis
- Recommended services
- Business recommendations
- Clustering model information
Dashboard screenshot:
images/dashboard.png

📈 Business Recommendations
The customer segmentation system can support organizations in:
1. Personalized Marketing
Different customer segments can receive different offers and communication strategies.
2. Customer Retention
Low-engagement customers can be targeted using re-engagement campaigns.
3. Premium Customer Management
High-value customers can receive premium services and loyalty benefits.
4. Learning Personalization
Customers with high learning interest can be directed toward relevant courses and certification programs.
5. Customer Support Prioritization
Customers with frequent service inquiries can be provided with proactive and priority support.
6. Data-Driven Decision Making
Customer segmentation allows business teams to make decisions based on measurable behavioral patterns rather than assumptions.
🧰 Technologies Used
Technology	Purpose
Python	Core programming language
Pandas	Data manipulation
NumPy	Numerical computing
Scikit-learn	Machine learning
K-Means	Customer clustering
StandardScaler	Feature standardization
PCA	Dimensionality reduction
Matplotlib	Data visualization
Seaborn	Statistical visualization
Streamlit	Interactive dashboard
Git	Version control
GitHub	Project repository


📁 Project Structure
Customer_Segmentation_Service_Recommendation/
│
├── data/
│   ├── customer_data.csv
│   └── customer_segmented_data.csv
│
├── images/
│   ├── elbow_curve.png
│   ├── silhouette_score.png
│   ├── cluster_visualization.png
│   ├── dashboard.png
│   └── rollbackevidence.png
│
├── notebooks/
│   └── Customer_Segmentation.ipynb
│
├── app.py
├── prepare_data.py
├── requirements.txt
├── deployment.yaml
├── rollback_evidence.md
├── README.md
└── .gitignore

🚀 How to Run the Project
Step 1 — Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>

Move into the project directory:
cd Customer_Segmentation_Service_Recommendation

Step 2 — Install Dependencies
pip install -r requirements.txt

Step 3 — Run the Streamlit Application
python -m streamlit run app.py

The dashboard will be available locally at:
http://localhost:8501

🔄 Data Preparation
The prepare_data.py script performs the complete data preparation and machine learning pipeline.
It includes:
- Feature engineering
- Feature scaling
- Cluster evaluation
- K-Means clustering
- Cluster profiling
- Segment naming
- Recommendation generation
- PCA transformation
- Final dataset generation
Run:
python prepare_data.py

🔐 Data Considerations
The dataset used in this project is synthetic and does not contain real customer personally identifiable information.
For production deployment, appropriate data governance, privacy, security and access-control mechanisms should be implemented.
🔁 Version Control & Rollback
Git was used to maintain project versions and preserve a stable release.
The stable version is tagged as:
v1.0

View available versions:
git tag

View version history:
git log --oneline --decorate

To inspect the stable version:
git checkout v1.0

The rollback procedure and evidence are documented in:
rollback_evidence.md

☁️ Deployment
A deployment configuration is included in:
deployment.yaml

The application can be containerized and deployed to a suitable cloud or Kubernetes-based environment.
The Streamlit application exposes port:
8501

📌 Key Project Outcomes
The project successfully demonstrates an end-to-end Data Science workflow:
Data Collection
      ↓
Data Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Feature Scaling
      ↓
Unsupervised Machine Learning
      ↓
Cluster Evaluation
      ↓
Customer Segmentation
      ↓
Recommendation System
      ↓
Interactive Dashboard
      ↓
Business Insights

🔮 Future Enhancements
The project can be further improved by:
- Using real-world customer datasets
- Adding automated model retraining
- Comparing K-Means with DBSCAN and hierarchical clustering
- Developing a personalized recommendation model
- Adding customer lifetime value prediction
- Integrating a database
- Adding authentication and role-based access
- Deploying the application to a cloud platform
- Adding monitoring and model drift detection
- Incorporating real-time customer behavior data
