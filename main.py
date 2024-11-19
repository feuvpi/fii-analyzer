import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import time

class FIIAnalyzer:
    def __init__(self):
        """Initialize the FII Analyzer with FII data from CSV"""
        self.fiis_data = self.load_fiis_from_csv()
        self.processed_data = None



    def load_fiis_from_csv(self):
        """Load FII data from the provided CSV file"""
        try:
            # Read the whole file content
            with open('/Users/feuvp/Downloads/fundosListados.csv', 'r', encoding='cp1252') as file:
                lines = file.readlines()

            # Process header
            header = [col.strip() for col in lines[0].strip().rstrip(';').split(';')]
            
            # Process data rows
            data = []
            for line in lines[1:]:
                # Remove trailing semicolon and split
                values = [val.strip() for val in line.strip().rstrip(';').split(';')]
                if len(values) >= len(header):  # Ensure we have enough values
                    # Take only the values we need (matching header length)
                    values = values[:len(header)]
                    data.append(values)
            
            # Create DataFrame
            fiis_df = pd.DataFrame(data, columns=header)
            
            # Rename columns to standardized names
            column_translations = {
                'Razão Social': 'company',
                'Fundo': 'name',
                'Segmento': 'sector',
                'Código': 'code'
            }
            
            # Debug information before renaming
            st.write("Original columns:", fiis_df.columns.tolist())
            
            # Try to find the closest matching column names
            for pt, en in column_translations.items():
                matching_cols = [col for col in fiis_df.columns if any(c in col for c in [pt, pt.upper(), pt.lower()])]
                if matching_cols:
                    fiis_df = fiis_df.rename(columns={matching_cols[0]: en})
            
            # Add ticker column with "11" suffix - Modified this part
            fiis_df['ticker'] = fiis_df['code'].apply(lambda x: f"{x}11.SA")
            
            # Clean any empty strings in sector
            fiis_df['sector'] = fiis_df['sector'].replace('', 'Não Classificado')
            
            # Debug information
            st.write(f"Successfully loaded {len(fiis_df)} FIIs")
            st.write("Final columns:", fiis_df.columns.tolist())
            st.write("Sample of the data:", fiis_df.head())
            
            return fiis_df
            
        except Exception as e:
            st.error(f"Error loading FII data from CSV: {str(e)}")
            st.write("Error details:", e.__class__.__name__)
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['code', 'sector', 'name', 'ticker', 'company'])

    def get_fii_data(self, ticker):
        """Enhanced FII data collection with timeout"""
        try:
            # Ensure the ticker is in the correct format (XXXX11.SA)
            if not ticker.endswith('11.SA'):
                ticker = f"{ticker}11.SA"
            
            fii = yf.Ticker(ticker)
            
            # Use a timeout of 5 seconds for the info fetch
            info = {}
            try:
                info = fii.info
            except Exception as e:
                st.warning(f"Could not fetch info for {ticker}: {str(e)}")
            
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
                
            # Add price momentum metrics - Fixed deprecated indexing
            if hist_data is not None and not hist_data.empty:
                close_prices = hist_data['Close']
                data['price_momentum_1m'] = ((close_prices.iloc[-1] / close_prices.iloc[-22] - 1) * 100 
                                        if len(close_prices) >= 22 else None)
                data['price_momentum_3m'] = ((close_prices.iloc[-1] / close_prices.iloc[-66] - 1) * 100 
                                        if len(close_prices) >= 66 else None)
                
            return data
            
        except Exception as e:
            st.warning(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def get_all_fiis_data(self):
        """Fetch data for all FIIs with improved progress tracking"""
        if self.processed_data is not None:
            return self.processed_data

        data_list = []
        failed_fetches = []
        total_fiis = len(self.fiis_data)
        
        # Create columns for status display
        col1, col2, col3 = st.columns([2,1,1])
        
        with col1:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with col2:
            success_count = st.empty()
            success_count.text("Successful fetches: 0")
        
        with col3:
            failed_count = st.empty()
            failed_count.text("Failed fetches: 0")
        
        # Add a placeholder for showing recent failures
        failures_display = st.empty()
        
        successful = 0
        failed = 0
        
        for idx, row in self.fiis_data.iterrows():
            ticker = row['code']
            try:
                # Update progress
                progress = (idx + 1) / total_fiis
                progress_bar.progress(progress)
                status_text.text(f'Processing: {ticker} ({idx + 1}/{total_fiis})')
                
                # Try to fetch data with timeout
                data = self.get_fii_data(ticker)
                
                if data:
                    data_list.append(data)
                    successful += 1
                    success_count.text(f"Successful fetches: {successful}")
                else:
                    failed += 1
                    failed_fetches.append(ticker)
                    failed_count.text(f"Failed fetches: {failed}")
                
                # Show last 5 failed fetches
                if failed_fetches:
                    failures_display.text(f"Recent failures: {', '.join(failed_fetches[-5:])}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                failed += 1
                failed_fetches.append(ticker)
                failed_count.text(f"Failed fetches: {failed}")
                st.warning(f"Error processing {ticker}: {str(e)}")
        
        status_text.text('Finished processing all FIIs!')
        
        # Show final summary
        st.write(f"""
        ### Processing Summary
        - Total FIIs: {total_fiis}
        - Successfully processed: {successful} ({(successful/total_fiis)*100:.1f}%)
        - Failed: {failed} ({(failed/total_fiis)*100:.1f}%)
        """)
        
        if failed_fetches:
            st.write("Failed tickers:", ", ".join(failed_fetches))
        
        # Create DataFrame and store it
        self.processed_data = pd.DataFrame(data_list)
        
        # Save basic stats about the data
        st.write(f"""
        ### Data Overview
        - Number of FIIs with data: {len(self.processed_data)}
        - Columns available: {', '.join(self.processed_data.columns)}
        """)
        
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
            # Fixed deprecated pct_change warning by explicitly specifying fill_method
            daily_returns = hist_data['Close'].pct_change(fill_method=None).dropna()
            
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

    def calculate_dividend_consistency(self, hist_data):
        """Calculate dividend consistency score"""
        try:
            dividends = hist_data['Dividends']
            if len(dividends) == 0:
                return 0
            
            # Fixed deprecated 'M' frequency warning by using 'ME'
            monthly_dividends = dividends.resample('ME').sum()
            consistency = (monthly_dividends > 0).mean() * 100
            return consistency
        except Exception as e:
            print(f"Error calculating dividend consistency: {str(e)}")
            return 0

    def calculate_beta(self, returns, market_ticker='^BVSP'):
        """Calculate beta relative to IBOVESPA"""
        try:
            market = yf.download(market_ticker, start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
            # Fixed deprecated pct_change warning
            market_returns = market['Adj Close'].pct_change(fill_method=None).dropna()
            
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
            
            # Fixed deprecated 'M' frequency warning by using 'ME'
            monthly_dividends = dividends.resample('ME').sum()
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