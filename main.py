import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io

class FIIAnalyzer:
    def __init__(self):
        """Initialize the FII Analyzer with FII data from CSV"""
        self.fiis_data = self.load_fiis_from_csv()
        self.processed_data = None

    def load_fiis_from_csv(self):
        """Load FII data from the provided CSV file with proper column handling"""
        try:
            # Common encodings for Portuguese files
            encodings = ['utf-8']
            
            for encoding in encodings:
                try:
                    # First try to peek at the file to determine the correct separator
                    with open('/Users/feuvp/Downloads/fundosListados.csv', 'r', encoding=encoding) as f:
                        first_line = f.readline().strip()
                        st.write("First line:", first_line)
                        
                        # If we find a semicolon in the first line, use that as separator
                        if ';' in first_line:
                            separator = ';'
                        else:
                            separator = ','  # fallback to comma
                    
                    # Read the CSV with the determined separator
                    fiis_df = pd.read_csv(
                        '/Users/feuvp/Downloads/fundosListados.csv',
                        encoding=encoding,
                        sep=separator,
                        decimal=',',  # Handle Brazilian number format
                        thousands='.'  # Handle Brazilian number format
                    )
                    
                    # Debug information
                    st.write("Initial columns:", fiis_df.columns.tolist())
                    st.write("Initial shape:", fiis_df.shape)
                    
                    # If we only got one column, try to split it
                    if len(fiis_df.columns) == 1:
                        # Split the single column into multiple columns
                        first_col_name = fiis_df.columns[0]
                        if isinstance(first_col_name, str) and any(sep in first_col_name for sep in [';', ',']):
                            # Split the column names
                            new_columns = [col.strip() for col in first_col_name.replace('"', '').split(';')]
                            st.write("Detected columns after split:", new_columns)
                            
                            # Split the data
                            new_data = [row[0].split(';') for _, row in fiis_df.iterrows()]
                            fiis_df = pd.DataFrame(new_data, columns=new_columns)
                    
                    # Debug the resulting DataFrame
                    st.write("Columns after processing:", fiis_df.columns.tolist())
                    st.write("Sample data:", fiis_df.head())
                    
                    # Rename columns to standardized names
                    column_translations = {
                        'Razão Social': 'company',
                        'Fundo': 'name',
                        'Segmento': 'sector',
                        'Código': 'code'
                    }
                    
                    fiis_df = fiis_df.rename(columns=column_translations)
                    
                    # Add ticker column
                    if 'code' in fiis_df.columns:
                        fiis_df['ticker'] = fiis_df['code'] + '.SA'
                    
                    # Ensure all required columns exist
                    required_columns = ['code', 'sector', 'name', 'ticker', 'company']
                    for col in required_columns:
                        if col not in fiis_df.columns:
                            fiis_df[col] = None
                    
                    return fiis_df
                    
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.warning(f"Error with encoding {encoding}: {str(e)}")
                    continue
            
            raise ValueError("Could not properly read the CSV file with any encoding")
            
        except Exception as e:
            st.error(f"Error loading FII data from CSV: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['code', 'sector', 'name', 'ticker', 'company'])



    def get_fii_data(self, ticker):
        """Enhanced FII data collection"""
        try:
            fii = yf.Ticker(ticker)
            info = fii.info
            hist_data = self.get_historical_data(ticker)
            risk_metrics = self.calculate_risk_metrics(hist_data)
            
            # Get sector information from fiis_data
            fii_info = self.fiis_data[self.fiis_data['ticker'] == ticker]
            sector = fii_info['sector'].iloc[0] if not fii_info.empty else 'Unknown'
            
            # Basic information
            data = {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'sector': sector,
                'price': info.get('regularMarketPrice', None),
                'previous_close': info.get('previousClose', None),
                'volume': info.get('volume', None),
                'market_cap': info.get('marketCap', None),
                'dividend_yield': info.get('dividendYield', None) * 100 if info.get('dividendYield') else None,
            }
            
            # Add risk metrics
            if risk_metrics:
                data.update(risk_metrics)
            
            # Calculate P/VP
            if 'bookValue' in info and data['price']:
                data['p_vp'] = data['price'] / info['bookValue']
            else:
                data['p_vp'] = None
                
            # Calculate dividend consistency
            if hist_data is not None:
                data['dividend_consistency'] = self.calculate_dividend_consistency(hist_data)
                
            # Add price momentum metrics
            if hist_data is not None and not hist_data.empty:
                data['price_momentum_1m'] = (hist_data['Close'][-1] / hist_data['Close'][-22] - 1) * 100 if len(hist_data) >= 22 else None
                data['price_momentum_3m'] = (hist_data['Close'][-1] / hist_data['Close'][-66] - 1) * 100 if len(hist_data) >= 66 else None
                
            return data
        except Exception as e:
            st.warning(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def get_all_fiis_data(self):
        """Fetch data for all FIIs"""
        if self.processed_data is not None:
            return self.processed_data

        data_list = []
        total_fiis = len(self.fiis_data)
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in self.fiis_data.iterrows():
            # Update progress
            progress = (idx + 1) / total_fiis
            progress_bar.progress(progress)
            status_text.text(f'Processing FII {idx + 1} of {total_fiis}: {row["ticker"]}')
            
            data = self.get_fii_data(row['ticker'])
            if data:
                data_list.append(data)
        
        status_text.text('Finished processing all FIIs!')
        progress_bar.empty()
        
        # Create DataFrame and store it
        self.processed_data = pd.DataFrame(data_list)
        return self.processed_data

    def get_historical_data(self, ticker, period='1y'):
        """Fetch historical data for analysis"""
        try:
            fii = yf.Ticker(ticker)
            hist = fii.history(period=period)
            return hist
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {str(e)}")
            return None

    def calculate_risk_metrics(self, hist_data):
        """Calculate risk-related metrics"""
        if hist_data is None or hist_data.empty:
            return {
                'volatility': None,
                'sharpe_ratio': None,
                'max_drawdown': None,
                'beta': None
            }
        
        try:
            daily_returns = hist_data['Close'].pct_change().dropna()
            
            return {
                'volatility': daily_returns.std() * np.sqrt(252) * 100,
                'sharpe_ratio': (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() != 0 else 0,
                'max_drawdown': ((hist_data['Close'] / hist_data['Close'].expanding(min_periods=1).max()) - 1).min() * 100,
                'beta': self.calculate_beta(daily_returns)
            }
        except Exception as e:
            print(f"Error calculating risk metrics: {str(e)}")
            return {
                'volatility': None,
                'sharpe_ratio': None,
                'max_drawdown': None,
                'beta': None
            }

    def calculate_beta(self, returns, market_ticker='^BVSP'):
        """Calculate beta relative to IBOVESPA"""
        try:
            market = yf.download(market_ticker, start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
            market_returns = market['Adj Close'].pct_change().dropna()
            
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) > 0:
                returns = returns[common_dates]
                market_returns = market_returns[common_dates]
                
                slope, _, _, _, _ = stats.linregress(market_returns, returns)
                return slope
        except Exception as e:
            print(f"Error calculating beta: {str(e)}")
            return None

    def calculate_dividend_consistency(self, hist_data):
        """Calculate dividend consistency score"""
        try:
            dividends = hist_data['Dividends']
            if len(dividends) == 0:
                return 0
            
            monthly_dividends = dividends.resample('M').sum()
            consistency = (monthly_dividends > 0).mean() * 100
            return consistency
        except Exception as e:
            print(f"Error calculating dividend consistency: {str(e)}")
            return 0

def create_streamlit_app():
    st.set_page_config(layout="wide")
    st.title("Análise de Fundos Imobiliários - FII da Bolsa de Valores do Brasil")
    
    # Initialize analyzer
    analyzer = FIIAnalyzer()
    
    # Add a refresh button
    if st.button("Refresh FII Data"):
        st.session_state.fii_data = None
    
    # Get or load data
    if 'fii_data' not in st.session_state or st.session_state.fii_data is None:
        with st.spinner('Fetching FII data...'):
            st.session_state.fii_data = analyzer.get_all_fiis_data()

    df = st.session_state.fii_data
    
    if df.empty:
        st.error("No data available. Please check the CSV file and try again.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Ensure we have valid sectors before creating the multiselect
    available_sectors = df['sector'].dropna().unique()
    
    # Basic filters
    selected_sectors = st.sidebar.multiselect(
        "Sectors",
        options=available_sectors,
        default=available_sectors
    )
    
    min_dy = st.sidebar.slider("Minimum Dividend Yield (%)", 0.0, 20.0, 0.0)
    max_p_vp = st.sidebar.slider("Maximum P/VP", 0.0, 3.0, 2.0)
    
    # Advanced filters
    st.sidebar.header("Advanced Filters")
    min_dividend_consistency = st.sidebar.slider("Minimum Dividend Consistency (%)", 0.0, 100.0, 50.0)
    max_volatility = st.sidebar.slider("Maximum Volatility (%)", 0.0, 50.0, 30.0)
    
    # Filter data
    filtered_df = df[
        (df['dividend_yield'].fillna(0) >= min_dy) &
        (df['p_vp'].fillna(float('inf')) <= max_p_vp) &
        (df['sector'].isin(selected_sectors)) &
        (df['dividend_consistency'].fillna(0) >= min_dividend_consistency) &
        (df['volatility'].fillna(float('inf')) <= max_volatility)
    ]

    

    # Rest of your code remains the same...
    # (Previous visualization code and tabs remain unchanged)

if __name__ == "__main__":
    create_streamlit_app()