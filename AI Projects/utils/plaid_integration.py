# utils/plaid_integration.py

import plaid
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from datetime import datetime, timedelta

def fetch_transactions(access_token):
    """Fetch transactions from Plaid API."""
    configuration = plaid.Configuration(
        host=plaid.Environment.Sandbox,  # Use Development or Production for real data
        api_key={
            "clientId": "your_plaid_client_id",
            "secret": "your_plaid_secret",
        }
    )
    api_client = plaid.ApiClient(configuration)
    client = plaid_api.PlaidApi(api_client)
    
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    request = TransactionsGetRequest(
        access_token=access_token,
        start_date=start_date,
        end_date=end_date,
        options=TransactionsGetRequestOptions()
    )
    response = client.transactions_get(request)
    return response.to_dict()["transactions"]