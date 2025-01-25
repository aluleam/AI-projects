# utils/visualization.py

import plotly.express as px
import plotly.graph_objects as go

def create_spending_chart(spending_by_category):
    """Create a bar chart for spending by category."""
    categories = list(spending_by_category.keys())
    amounts = list(spending_by_category.values())
    
    fig = px.bar(x=categories, y=amounts, labels={"x": "Category", "y": "Amount"}, title="Spending by Category")
    return fig.to_html(full_html=False)

def create_budget_chart(total_spending, budget):
    """Create a pie chart for budget tracking."""
    remaining_budget = max(0, budget - total_spending)
    fig = go.Figure(data=[go.Pie(labels=["Spent", "Remaining"], values=[total_spending, remaining_budget])])
    fig.update_layout(title="Budget Tracking")
    return fig.to_html(full_html=False)

def create_monthly_trends_chart(monthly_spending):
    """Create a line chart for monthly spending trends."""
    months = list(monthly_spending.keys())
    amounts = list(monthly_spending.values())
    
    fig = px.line(x=months, y=amounts, labels={"x": "Month", "y": "Amount"}, title="Monthly Spending Trends")
    return fig.to_html(full_html=False)