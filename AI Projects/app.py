# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from utils.data_processing import analyze_spending, provide_financial_advice, cluster_transactions
from utils.visualization import create_spending_chart, create_budget_chart, create_monthly_trends_chart
from utils.plaid_integration import fetch_transactions
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)

# Mock user database
class User(UserMixin):
    def __init__(self, id):
        self.id = id

users = {"1": User("1")}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    user_id = request.form.get("user_id")
    user = users.get(user_id)
    if user:
        login_user(user)
        return redirect(url_for("dashboard"))
    flash("Invalid user ID")
    return redirect(url_for("home"))

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route("/dashboard")
@login_required
def dashboard():
    try:
        # Fetch transactions from Plaid (replace with real access token)
        access_token = "your_plaid_access_token"
        transactions = fetch_transactions(access_token)
        
        # Analyze spending
        analysis = analyze_spending(transactions)
        
        # Provide financial advice
        advice = provide_financial_advice(analysis)
        
        # Cluster transactions
        cluster_analysis = cluster_transactions(transactions)
        
        # Create visualizations
        spending_chart = create_spending_chart(analysis["spending_by_category"])
        budget_chart = create_budget_chart(analysis["total_spending"], 5000)  # Example budget
        monthly_trends_chart = create_monthly_trends_chart(analysis["monthly_spending"])
        
        return render_template(
            "dashboard.html",
            analysis=analysis,
            advice=advice,
            spending_chart=spending_chart,
            budget_chart=budget_chart,
            monthly_trends_chart=monthly_trends_chart,
        )
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for("home"))

@app.route("/budget")
@login_required
def budget():
    return render_template("budget.html")

if __name__ == "__main__":
    app.run(debug=True)