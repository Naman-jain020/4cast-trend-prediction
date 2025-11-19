# 🔮 4Cast - Product Trend Prediction System

<div align="center">

![4Cast Logo](https://img.shields.io/badge/4Cast-Trend_Prediction-orange?style=for-the-badge)
![Django](https://img.shields.io/badge/Django-4.2-green?style=flat-square&logo=django)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**Foresee future with 4Cast - Forecast Like Never Before**

A powerful Django-based web application that fetches real-time data from Google Trends and predicts product growth trends using advanced time series forecasting and AI-powered insights.

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack) • [Contributing](#-contributing)

</div>

---

## 🌟 Features

### Core Capabilities
- 📊 **Real-time Google Trends Data** - Fetch up-to-date search interest data for any product/keyword
- 📈 **Time Series Forecasting** - Advanced prediction using Exponential Smoothing algorithm
- 🔄 **Multi-Keyword Comparison** - Compare trends for multiple products simultaneously
- 📁 **CSV Upload Support** - Analyze your own historical data
- 🎨 **Interactive Visualizations** - Beautiful, interactive Plotly graphs with dark theme
- 🤖 **AI-Powered Insights** - Get detailed business analysis using Google Gemini AI
- 💬 **Personalized Chatbot** - Ask specific questions about your data
- 🛒 **Smart Product Links** - Direct Amazon product search integration
- 💾 **Intelligent Caching** - Faster subsequent queries with local data caching

### Advanced Features
- ⚡ Rate limiting and retry logic for API stability
- 🔐 Secure environment variable management
- 📱 Responsive design for all devices
- 🎯 Peak demand detection with visual annotations
- ⏱️ Customizable forecast periods

---

## 🎬 Demo

### Homepage
<p align="center">
  <img src="screenshots/homepage.png" alt="4Cast Homepage" width="700"/>
</p>

### Results & Prediction Graph
<p align="center">
  <img src="screenshots/results.png" alt="Prediction Results" width="700"/>
</p>

> **Live Demo:** [Add your deployment link here]

---

## 🛠️ Tech Stack

### Backend
- **Django 4.2** - Web framework
- **PyTrends** - Google Trends API wrapper
- **Pandas & NumPy** - Data manipulation
- **Statsmodels** - Time series forecasting (Exponential Smoothing)

### Frontend & Visualization
- **Plotly** - Interactive data visualization
- **HTML/CSS/JavaScript** - Responsive UI
- **Markdown** - Rich text rendering

### AI & ML
- **Google Generative AI (Gemini 2.5 Flash)** - AI-powered insights
- **Exponential Smoothing** - Forecasting algorithm

### Others
- **Python Decouple** - Environment management
- **urllib3** - HTTP requests handling

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git
- Google Gemini API key ([Get one here](https://ai.google.dev/))

### Step 1: Clone the Repository
```
git clone https://github.com/yourusername/4cast-trend-prediction.git cd 4cast-trend-prediction
```
### Step 2: Create Virtual Environment
```
python3 -m venv venv 
source venv/bin/activate
```
### Step 3: Install Dependencies
```
pip install -r requirements.txt
```
### Step 4: Set Up Environment Variables
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
SECRET_KEY=your_django_secret_key_here
DEBUG=True
```
### Step 5: Run Migrations
```
python manage.py migrate
```
### Step 6: Run Development Server
```
python manage.py runserver
```


Visit: `http://127.0.0.1:8000/`

---

## 🚀 Usage

### Basic Workflow

1. **Enter Keywords**
   - Type product names/keywords (e.g., "iPhone", "Samsung Galaxy")
   - Add multiple keywords to compare trends

2. **Set Forecast Period**
   - Enter number of weeks to predict (e.g., 30 weeks)

3. **Optional: Upload CSV**
   - Format: First column = Date, Second column = Interest values
   - Use if you have custom historical data

4. **Ask Personalized Questions**
   - Example: "Which is the most trending iPhone currently in the market?"
   - Get AI-powered answers specific to your data

5. **Get Insights**
   - Click "Get Insights" button
   - View interactive graph with historical and forecasted data
   - Read detailed AI analysis

