from django.shortcuts import render
from django.http import HttpResponse
from pytrends.request import TrendReq
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
import plotly.graph_objs as go
import numpy as np
import google.generativeai as genai
import markdown
import os
import json
import time
import random
from decouple import config

# Define the path to the storage file
STORAGE_FILE = 'keyword_data.json'

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def feedback(request):
    return render(request, 'feedback.html')

def faq(request):
    return render(request, 'faq.html')

def trending(request):
    return render(request,'trending.html')

def save_feedback(request):
    if request.method == 'POST':
        return HttpResponse(status=200)
    else:
        return HttpResponse(status=405)

def save_to_text_file(request):
    if request.method == 'POST':
        try:
            save_directory = os.path.join(os.path.dirname(__file__), 'Customer_Data')
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            data = request.body.decode('utf-8')
            with open(os.path.join(save_directory, 'data.txt'), 'a') as f:
                f.write(data + '\n')

            return HttpResponse(status=200)
        except Exception as e:
            print(e)
            return HttpResponse(status=500)
    else:
        return HttpResponse(status=405)

def load_data_from_file(keyword):
    """Load cached data from file if it exists"""
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, 'r') as f:
                stored_data = json.load(f)
            return stored_data.get(keyword)
        except Exception as e:
            print(f"Error loading cached  {e}")
            return None
    return None

def save_data_to_file(keyword, df):
    """Save fetched data to file for future use"""
    try:
        if os.path.exists(STORAGE_FILE):
            with open(STORAGE_FILE, 'r') as f:
                stored_data = json.load(f)
        else:
            stored_data = {}
        
        data_to_save = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'interest': df['interest'].tolist()
        }
        stored_data[keyword] = data_to_save
        
        with open(STORAGE_FILE, 'w') as f:
            json.dump(stored_data, f)
    except Exception as e:
        print(f"Error saving data to file: {e}")

def get_pytrends_instance():
    """Create a properly configured TrendReq instance with enhanced headers"""
    # Rotate user agents to avoid detection
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]
    
    selected_ua = random.choice(user_agents)
    
    headers = {
        'User-Agent': selected_ua,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'DNT': '1'
    }
    
    # Create TrendReq with enhanced configuration
    pytrends = TrendReq(
        hl='en-US',
        tz=360,
        timeout=(15, 90),  # Increased timeout
        retries=5,  # More retries
        backoff_factor=1.0,  # Longer backoff
        requests_args={
            'headers': headers,
            'verify': True
        }
    )
    
    return pytrends

def fetch_google_trends_data(pytrends, keyword, retries=8):
    """Fetch Google Trends data with enhanced retry logic"""
    for i in range(retries):
        try:
            # Progressive delay strategy
            if i == 0:
                wait_time = random.uniform(2, 4)
            else:
                wait_time = min((2 ** i) + random.uniform(3, 7), 60)  # Cap at 60 seconds
            
            print(f"Attempt {i + 1}/{retries} for {keyword}. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            
            # Recreate instance every 2 attempts to get fresh session
            if i > 0 and i % 2 == 0:
                print(f"Recreating pytrends instance for fresh session...")
                pytrends = get_pytrends_instance()
            
            # Build payload and fetch data
            pytrends.build_payload(kw_list=[keyword], timeframe='today 5-y', geo='IN')
            df = pytrends.interest_over_time()
            
            if not df.empty:
                print(f"✓ Successfully fetched data for keyword: {keyword}")
                return df
            else:
                print(f"⚠ Empty dataframe returned for keyword: {keyword}")
                
        except Exception as e:
            error_str = str(e)
            print(f"✗ Error on attempt {i + 1} for {keyword}: {error_str}")
            
            # Handle different error types
            if '429' in error_str:
                wait_time = min((3 ** i) + random.uniform(5, 10), 120)
                print(f"⚠ Rate limit (429). Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            elif '500' in error_str or '503' in error_str:
                wait_time = random.uniform(5, 10)
                print(f"⚠ Server error. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"⚠ Unexpected error: {error_str}")
                time.sleep(random.uniform(3, 6))
    
    raise Exception(f"Failed to fetch data for keyword: {keyword} after {retries} retries. Google Trends is blocking requests. Please try: 1) Using cached data 2) Waiting a few hours 3) Using a VPN/different network")

def complex_graph_data_to_text(graph_data):
    """Convert graph data to text description for AI analysis"""
    description = f"Graph titled '{graph_data['title']}' showing data over {graph_data['x']} with values on {graph_data['y']}. "
    description += "Data points are: " + ", ".join([f"({date}, {value})" for date, value in graph_data['historical_data'][:10]])
    if len(graph_data['historical_data']) > 10:
        description += f" ... and {len(graph_data['historical_data']) - 10} more points"
    description += f". Forecasted  " + ", ".join([f"({date}, {value:.2f})" for date, value in graph_data['forecasted_data'][:5]])
    if len(graph_data['forecasted_data']) > 5:
        description += f" ... and {len(graph_data['forecasted_data']) - 5} more points"
    description += f". Forecast period: {graph_data['forecast_periods']} weeks."
    return description

def predict(request):
    if request.method == 'POST':
        keywords = request.POST.getlist('keywords')
        forecast_periods = int(request.POST.get('forecast_periods', 0))
        csv_file = request.FILES.get('csv_file')
        user_query = request.POST.get('user_query', '')

        historical_data = {}
        forecast_data = {}
        graph_descriptions = []
        failed_keywords = []

        # Process uploaded CSV file if provided
        if csv_file:
            try:
                df = pd.read_csv(csv_file)
                
                first_column = df.columns[0]
                df[first_column] = pd.to_datetime(df[first_column], dayfirst=True)
                df.set_index(first_column, inplace=True)
                keyword = df.columns[0]
                df = df.rename(columns={keyword: 'interest'})

                model = ExponentialSmoothing(df['interest'], trend='add', seasonal='add', seasonal_periods=52)
                fit_model = model.fit()
                forecast = fit_model.forecast(steps=forecast_periods)

                df['interest'] = np.maximum(df['interest'], 0)
                forecast = np.maximum(forecast, 0)

                df.index = df.index.date
                forecast_index = pd.date_range(start=df.index[-1], periods=forecast_periods + 1, freq='W')[1:].date

                historical_data[keyword] = df
                forecast_data[keyword] = pd.Series(forecast, index=forecast_index)

                historical_data_points = [(date.strftime('%Y/%m/%d'), value) for date, value in zip(df.index.tolist(), df['interest'].tolist())]
                forecasted_data_points = [(date.strftime('%Y/%m/%d'), value) for date, value in zip(forecast_index.tolist(), forecast.tolist())]

                graph_data = {
                    'title': keyword,
                    'x': 'Date',
                    'y': 'Interest',
                    'historical_data': historical_data_points,
                    'forecasted_data': forecasted_data_points,
                    'forecast_periods': forecast_periods
                }
                graph_descriptions.append(complex_graph_data_to_text(graph_data))
            except Exception as e:
                return HttpResponse(f"Error processing CSV file: {str(e)}")

        # Process keywords if provided
        if keywords:
            pytrends = get_pytrends_instance()
            
            # Add initial delay before first request
            print("Initializing connection to Google Trends...")
            time.sleep(random.uniform(3, 5))
            
            for idx, keyword in enumerate(keywords):
                keyword = keyword.strip()
                print(f"\n{'='*50}")
                print(f"Processing keyword {idx + 1}/{len(keywords)}: '{keyword}'")
                print(f"{'='*50}")
                
                try:
                    # Check cache first
                    cached_data = load_data_from_file(keyword)
                    if cached_data:
                        print(f"✓ Using cached data for keyword: {keyword}")
                        historical_dates = [pd.to_datetime(date).date() for date in cached_data['dates']]
                        historical_interest = cached_data['interest']
                        df = pd.DataFrame({'interest': historical_interest}, index=pd.to_datetime(historical_dates))
                    else:
                        print(f"⚠ No cached data found. Fetching from Google Trends...")
                        # Fetch from Google Trends
                        df_raw = fetch_google_trends_data(pytrends, keyword)
                        df = df_raw.drop(['isPartial'], axis=1, errors='ignore')
                        df = df.rename(columns={keyword: 'interest'})
                        
                        # Save to cache
                        save_data_to_file(keyword, df)
                        print(f"✓ Saved data to cache for keyword: {keyword}")
                    
                    # Fit model and forecast
                    model = ExponentialSmoothing(df['interest'], trend='add', seasonal='add', seasonal_periods=52)
                    fit_model = model.fit()
                    forecast = fit_model.forecast(steps=forecast_periods)

                    df['interest'] = np.maximum(df['interest'], 0)
                    forecast = np.maximum(forecast, 0)

                    df.index = df.index.date
                    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_periods + 1, freq='W')[1:].date

                    historical_data[keyword] = df
                    forecast_data[keyword] = pd.Series(forecast, index=forecast_index)

                    historical_data_points = [(date.strftime('%Y/%m/%d'), value) for date, value in zip(df.index.tolist(), df['interest'].tolist())]
                    forecasted_data_points = [(date.strftime('%Y/%m/%d'), value) for date, value in zip(forecast_index.tolist(), forecast.tolist())]

                    graph_data = {
                        'title': keyword,
                        'x': 'Date',
                        'y': 'Interest',
                        'historical_data': historical_data_points,
                        'forecasted_data': forecasted_data_points,
                        'forecast_periods': forecast_periods
                    }
                    graph_descriptions.append(complex_graph_data_to_text(graph_data))
                    
                    print(f"✓ Successfully processed keyword: {keyword}")
                    
                    # Add delay between keywords (longer for non-cached)
                    if idx < len(keywords) - 1:
                        if cached_data:
                            delay = random.uniform(1, 2)
                        else:
                            delay = random.uniform(10, 15)  # Much longer delay after fresh fetch
                        print(f"⏳ Waiting {delay:.2f} seconds before next keyword...")
                        time.sleep(delay)
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"✗ Error processing keyword '{keyword}': {error_message}")
                    failed_keywords.append(keyword)
                    
                    # Don't fail completely - continue with other keywords
                    if idx < len(keywords) - 1:
                        print(f"⚠ Continuing with remaining keywords...")
                        time.sleep(random.uniform(5, 8))
                    continue

        # Check if any data was successfully loaded
        if not historical_data:
            error_msg = "No data was successfully fetched or loaded. "
            if failed_keywords:
                error_msg += f"Failed keywords: {', '.join(failed_keywords)}. "
            error_msg += "\n\nPossible solutions:\n"
            error_msg += "1. Try again in a few hours (Google may have temporarily blocked your IP)\n"
            error_msg += "2. Use a VPN or different network connection\n"
            error_msg += "3. Upload a CSV file instead of fetching from Google Trends\n"
            error_msg += "4. The data might be cached - try searching for keywords you've used before"
            return HttpResponse(error_msg)

        # Show warning if some keywords failed
        warning_message = ""
        if failed_keywords:
            warning_message = f"⚠ Warning: Failed to fetch data for: {', '.join(failed_keywords)}. Showing results for successful keywords only."
            print(f"\n{warning_message}")

        combined_graph_description = "\n\n".join(graph_descriptions)

        # Configure and use Google Generative AI
        genai.configure(api_key=config('GEMINI_API_KEY'))

        model = genai.GenerativeModel('gemini-flash-latest')

        successful_keywords = list(historical_data.keys())
        prompt = f"You are provided with graph data representing the historical and forecasted interest in the products: {', '.join(successful_keywords)}. Your task is to analyze this data and provide detailed business insights, including trends, future possibilities, and considerations for different demographics. Tell all the basic data and other information to the user as business analyst. In conclusion section, be blunt as to whether you should invest in the product or not with reason. Be practical and explain like an expert. give proper conclusion whether its comparing or providing suggestions regarding the product/products. REMEMBER TO KEEP MOST OF THE BIASES OR VARIABLE IN MIND AT LEAST WHICH YOU CAN HELP WITH"

        if user_query:
            user_query_section = f"\nAfter conclusion please\nUser Query: {user_query}\n Please provide a specific answer to this query based on the data provided. DONT GIVE DIPLOMATIC ANSWERS SUGGEST WHAT YOU THINK"
            prompt += user_query_section

        combined_prompt = f"{prompt}\nGraph Data:\n{combined_graph_description}"
        
        try:
            response = model.generate_content(combined_prompt)
            summary_html = markdown.markdown(response.text)
            
            # Add warning to summary if some keywords failed
            if warning_message:
                summary_html = f"<div style='background-color: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin-bottom: 20px; border-radius: 5px;'><strong>⚠ {warning_message}</strong></div>" + summary_html
                
        except Exception as e:
            summary_html = f"<p>Error generating AI summary: {str(e)}</p>"

        # Plotting with Plotly
        traces = []
        annotations = []

        for keyword, historical_df in historical_data.items():
            forecast_series = forecast_data[keyword]

            traces.append(go.Scatter(
                x=historical_df.index,
                y=historical_df['interest'],
                mode='lines+markers',
                name=f'{keyword} (Historical)'
            ))
            traces.append(go.Scatter(
                x=forecast_series.index,
                y=forecast_series,
                mode='lines+markers',
                name=f'{keyword} (Forecast)',
                line=dict(dash='dash')
            ))

            peak_date = historical_df['interest'].idxmax()
            peak_value = historical_df['interest'].max()

            annotation_y = peak_value + 5 if len(annotations) % 2 == 0 else peak_value - 5

            annotations.append(dict(
                x=peak_date,
                y=annotation_y,
                xref='x',
                yref='y',
                text=f'{keyword} demand peak is here\n{peak_date.strftime("%Y-%m-%d")}',
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-80 if len(annotations) % 2 == 0 else 80,
                bgcolor='rgba(255, 0, 0, 0.7)',
                font=dict(size=14, color='white'),
                bordercolor='black',
                borderwidth=2,
                borderpad=4,
                opacity=0.8
            ))

        layout = go.Layout(
            title='Result Graph: ',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Interest'),
            template='plotly_dark',
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='#ffffff'),
            legend=dict(x=0.5, y=-0.2, xanchor='center', yanchor='top', orientation='h'),
            autosize=True,
            annotations=annotations
        )

        fig = go.Figure(data=traces, layout=layout)
        plot_div = fig.to_html(full_html=False, default_height='100%', default_width='100%')

        keyword_search_links = [(keyword, f"https://www.amazon.in/s?k={keyword}") for keyword in successful_keywords]

        return render(request, 'result.html', {
            'plot_div': plot_div,
            'summary_string': summary_html,
            'keyword_search_links': keyword_search_links
        })

    return render(request, 'index.html')
