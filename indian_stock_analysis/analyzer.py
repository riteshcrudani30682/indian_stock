# Moving technical_analysis_bot.py content here with modifications
import yfinance as yf
import pandas as pd
import numpy as np
import talib

class IndianStockAnalyzer:
    def __init__(self):
        # NIFTY 500 Stocks by Sector
        self.symbols = [
            # Content from technical_analysis_bot.py
            # ... (keeping the existing symbols list)
        ]
        
    def fetch_data(self, symbol, period='1y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        if df is None or df.empty:
            return None
        
        try:
            # RSI
            df['RSI'] = talib.RSI(df['Close'])
            
            # MACD
            df['MACD'], df['Signal'], df['Hist'] = talib.MACD(df['Close'])
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
            
            return df
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return None

    def generate_signals(self, df):
        """Generate trading signals based on indicators"""
        if df is None or df.empty:
            return None
        
        try:
            # RSI Signals
            df['RSI_Signal'] = 'Neutral'
            df.loc[df['RSI'] < 30, 'RSI_Signal'] = 'Oversold'
            df.loc[df['RSI'] > 70, 'RSI_Signal'] = 'Overbought'
            
            # MACD Signals
            df['MACD_Signal'] = 'Neutral'
            df.loc[(df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1)), 'MACD_Signal'] = 'Buy'
            df.loc[(df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1)), 'MACD_Signal'] = 'Sell'
            
            # Bollinger Bands Position
            df['BB_Position'] = 'Middle'
            df.loc[df['Close'] >= df['BB_Upper'], 'BB_Position'] = 'Upper'
            df.loc[df['Close'] <= df['BB_Lower'], 'BB_Position'] = 'Lower'
            
            return df
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return None

    def calculate_sector_performance(self, sector_symbols, period='1y'):
        """Calculate sector performance metrics"""
        sector_data = {
            'returns': [],
            'volatility': [],
            'strength': [],
            'volume': []
        }
        
        for symbol in sector_symbols:
            try:
                df = self.fetch_data(symbol, period)
                if df is not None and not df.empty:
                    # Calculate returns
                    returns = ((df['Close'][-1] - df['Close'][0]) / df['Close'][0]) * 100
                    sector_data['returns'].append(returns)
                    
                    # Calculate volatility
                    daily_returns = df['Close'].pct_change()
                    volatility = daily_returns.std() * 100
                    sector_data['volatility'].append(volatility)
                    
                    # Calculate relative strength
                    ma50 = df['Close'].rolling(window=50).mean()
                    strength = ((df['Close'][-1] - ma50[-1]) / ma50[-1]) * 100
                    sector_data['strength'].append(strength)
                    
                    # Calculate average volume
                    avg_volume = df['Volume'].mean()
                    sector_data['volume'].append(avg_volume)
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Calculate sector metrics
        if sector_data['returns']:
            sector_metrics = {
                'avg_return': np.mean(sector_data['returns']),
                'avg_volatility': np.mean(sector_data['volatility']),
                'avg_strength': np.mean(sector_data['strength']),
                'avg_volume': np.mean(sector_data['volume']),
                'best_performer': max(sector_data['returns']),
                'worst_performer': min(sector_data['returns']),
                'stocks_analyzed': len(sector_data['returns'])
            }
            return sector_metrics
        return None

    def get_all_sectors_performance(self, sectors_dict, period='1y'):
        """Calculate performance metrics for all sectors"""
        sectors_performance = {}
        for sector_name, symbols in sectors_dict.items():
            metrics = self.calculate_sector_performance(symbols, period)
            if metrics:
                sectors_performance[sector_name] = metrics
        return sectors_performance
