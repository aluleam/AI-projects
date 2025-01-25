# AI Projects Repository

This repository contains a collection of AI projects built to solve real-world problems. Below is a detailed overview of each project, setup instructions, and tools used.

---

## 📂 Projects

### 1. **Resume Parser**
**What it does**:  
Extracts structured data (skills, education, work experience) from PDF or DOCX resumes.  

**Tools Used**:  
- Python  
- PDFMiner (PDF text extraction)  
- spaCy (text processing)  

**Code Example**:  
```python
from pdfminer.high_level import extract_text

def extract_text_from_pdf(file_path):
    text = extract_text(file_path)
    return text

# Example usage
resume_text = extract_text_from_pdf("resume.pdf")
print("Extracted Text:", resume_text[:200])  # Show first 200 characters
```

---

### 2. **Disease Prediction**
**What it does**:  
Predicts potential diseases based on user-reported symptoms.  

**Tools Used**:  
- Python  
- scikit-learn  
- Pandas  

**Code Example**:  
```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
data = pd.read_csv("symptoms.csv")
X = data.drop("disease", axis=1)
y = data["disease"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Predict disease for new symptoms
new_symptoms = [[1, 0, 1, 0]]  # Example: [fever, cough, headache, fatigue]
prediction = model.predict(new_symptoms)
print("Predicted Disease:", prediction[0])
```

---

### 3. **Finance Assistant**
**What it does**:  
Helps users track expenses, analyze stocks, and manage budgets.  

**Tools Used**:  
- Python  
- yfinance  
- Streamlit  

**Code Example**:  
```python
import yfinance as yf

def get_stock_data(ticker, period="1mo"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Example usage
apple_data = get_stock_data("AAPL")
print("Apple Stock Data (Last Month):\n", apple_data.head())
```

---

### 4. **Sentiment Analysis**
**What it does**:  
Analyzes text (reviews, tweets) to detect positive, negative, or neutral sentiment.  

**Tools Used**:  
- Python  
- Hugging Face Transformers  

**Code Example**:  
```python
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result

# Example usage
result = analyze_sentiment("I loved the customer service!")
print("Sentiment:", result)  # Output: {'label': 'POSITIVE', 'score': 0.99}
```

---

## 🛠️ Tools & Technologies

### Core Tools
- **Python**: Primary programming language.  
- **scikit-learn**: Machine learning (Disease Prediction).  
- **spaCy**: Text processing (Resume Parser).  
- **Hugging Face Transformers**: NLP models (Sentiment Analysis).  

### Collaboration & Optimization
- **ChatGPT**: Code debugging and documentation.  
- **DeepSeek**: Data analysis support.  

---

## ⚙️ Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/AI-projects.git
   cd AI-projects
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Resume Parser
```bash
cd resume-parser
python main.py --filepath "resumes/sample.pdf"
```

### Disease Prediction
```bash
cd disease-prediction
python predict.py --symptoms "fever, cough, fatigue"
```

### Sentiment Analysis
```bash
cd sentiment-analysis
python app.py --text "The service was excellent!"
```
