
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify, render_template

# Step 2: Generate Mock Financial Data
def generate_mock_data():
    """Generate mock financial data for analysis."""
    np.random.seed(42)
    categories = ["Food", "Transport", "Entertainment", "Utilities", "Shopping", "Healthcare"]
    data = {
        "Date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
        "Category": np.random.choice(categories, 100),
        "Amount": np.random.randint(10, 500, 100),
    }
    df = pd.DataFrame(data)
    return df

# Step 3: Analyze Spending Habits
def analyze_spending(df):
    """Analyze spending habits and provide insights."""
    # Total spending
    total_spending = df["Amount"].sum()
    
    # Spending by category
    spending_by_category = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    
    # Average spending per transaction
    avg_spending = df["Amount"].mean()
    
    # Most frequent category
    most_frequent_category = df["Category"].mode()[0]
    
    return {
        "total_spending": total_spending,
        "spending_by_category": spending_by_category.to_dict(),
        "avg_spending": avg_spending,
        "most_frequent_category": most_frequent_category,
    }

# Step 4: Provide Financial Advice
def provide_financial_advice(analysis):
    """Provide financial advice based on spending analysis."""
    advice = []
    
    # Advice on total spending
    if analysis["total_spending"] > 5000:
        advice.append("Your total spending is high. Consider creating a budget to manage your expenses better.")
    else:
        advice.append("Your total spending is within a reasonable range. Keep it up!")
    
    # Advice on spending by category
    for category, amount in analysis["spending_by_category"].items():
        if amount > 1000:
            advice.append(f"You're spending a lot on {category}. Consider cutting back on this category.")
    
    # Advice on average spending
    if analysis["avg_spending"] > 100:
        advice.append("Your average spending per transaction is high. Look for ways to reduce unnecessary expenses.")
    
    return advice

# Step 5: Cluster Transactions for Insights
def cluster_transactions(df):
    """Cluster transactions to identify spending patterns."""
    # Convert categories to numerical values
    df["Category_Code"] = df["Category"].astype("category").cat.codes
    
    # Use KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df[["Category_Code", "Amount"]])
    
    # Analyze clusters
    cluster_analysis = df.groupby("Cluster")["Category"].value_counts().unstack().fillna(0)
    return cluster_analysis

# Step 6: Flask App for Deployment
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # Generate mock data
    df = generate_mock_data()
    
    # Analyze spending habits
    analysis = analyze_spending(df)
    
    # Provide financial advice
    advice = provide_financial_advice(analysis)
    
    # Cluster transactions
    cluster_analysis = cluster_transactions(df)
    
    # Prepare response
    response = {
        "analysis": analysis,
        "advice": advice,
        "cluster_analysis": cluster_analysis.to_dict(),
    }
    return jsonify(response)

# Step 7: Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)