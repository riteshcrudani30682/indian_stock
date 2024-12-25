import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndianStockAnalyzer:
    def __init__(self):
        # NIFTY 500 Stocks by Sector
        self.symbols = [
            # Large Cap Banks & NBFCs
            'HDFCBANK.NS',    # HDFC Bank
            'ICICIBANK.NS',   # ICICI Bank
            'SBIN.NS',        # State Bank of India
            'KOTAKBANK.NS',   # Kotak Mahindra Bank
            'AXISBANK.NS',    # Axis Bank
            'INDUSINDBK.NS',  # IndusInd Bank
            'BANKBARODA.NS',  # Bank of Baroda
            'PNB.NS',         # Punjab National Bank
            'FEDERALBNK.NS',  # Federal Bank
            'IDFCFIRSTB.NS',  # IDFC First Bank
            'AUBANK.NS',      # AU Small Finance Bank
            'BANDHANBNK.NS',  # Bandhan Bank
            
            # Financial Services & Insurance
            'BAJFINANCE.NS',  # Bajaj Finance
            'HDFCLIFE.NS',    # HDFC Life Insurance
            'SBILIFE.NS',     # SBI Life Insurance
            'ICICIGI.NS',     # ICICI Lombard
            'HDFCAMC.NS',     # HDFC AMC
            'MUTHOOTFIN.NS',  # Muthoot Finance
            'CHOLAFIN.NS',    # Cholamandalam Investment
            'LTIM.NS',        # L&T Finance Holdings
            'PFC.NS',         # Power Finance Corporation
            'RECLTD.NS',      # REC Limited
            
            # IT & Technology
            'TCS.NS',         # Tata Consultancy Services
            'INFY.NS',        # Infosys
            'WIPRO.NS',       # Wipro
            'HCLTECH.NS',     # HCL Technologies
            'TECHM.NS',       # Tech Mahindra
            'LTTS.NS',        # L&T Technology Services
            'PERSISTENT.NS',  # Persistent Systems
            'MPHASIS.NS',     # Mphasis
            'COFORGE.NS',     # Coforge
            'CYIENT.NS',      # Cyient
            'MINDTREE.NS',    # Mindtree
            
            # Oil & Gas
            'RELIANCE.NS',    # Reliance Industries
            'ONGC.NS',        # Oil & Natural Gas Corporation
            'IOC.NS',         # Indian Oil Corporation
            'BPCL.NS',        # Bharat Petroleum
            'GAIL.NS',        # GAIL India
            'HINDPETRO.NS',   # Hindustan Petroleum
            'PETRONET.NS',    # Petronet LNG
            'IGL.NS',         # Indraprastha Gas
            'MGL.NS',         # Mahanagar Gas
            'GUJGASLTD.NS',   # Gujarat Gas
            
            # Power & Energy
            'NTPC.NS',        # NTPC Limited
            'POWERGRID.NS',   # Power Grid Corporation
            'ADANIPOWER.NS',  # Adani Power
            'TATAPOWER.NS',   # Tata Power
            'TORNTPOWER.NS',  # Torrent Power
            'NHPC.NS',        # NHPC
            'SUZLON.NS',      # Suzlon Energy
            'ADANIGREEN.NS',  # Adani Green Energy
            'JSL.NS',         # Jindal Steel
            
            # Metals & Mining
            'TATASTEEL.NS',   # Tata Steel
            'HINDALCO.NS',    # Hindalco Industries
            'JSWSTEEL.NS',    # JSW Steel
            'SAIL.NS',        # Steel Authority of India
            'JINDALSTEL.NS',  # Jindal Steel & Power
            'NATIONALUM.NS',  # National Aluminium
            'COALINDIA.NS',   # Coal India
            'NMDC.NS',        # NMDC
            'VEDL.NS',        # Vedanta
            'HINDZINC.NS',    # Hindustan Zinc
            
            # Automobiles
            'TATAMOTORS.NS',  # Tata Motors
            'M&M.NS',         # Mahindra & Mahindra
            'MARUTI.NS',      # Maruti Suzuki
            'BAJAJ-AUTO.NS',  # Bajaj Auto
            'EICHERMOT.NS',   # Eicher Motors
            'ASHOKLEY.NS',    # Ashok Leyland
            'TVSMOTOR.NS',    # TVS Motor
            'HEROMOTOCO.NS',  # Hero MotoCorp
            'BALKRISIND.NS',  # Balkrishna Industries
            'MRF.NS',         # MRF
            'APOLLOTYRE.NS',  # Apollo Tyres
            
            # Consumer Goods
            'HINDUNILVR.NS',  # Hindustan Unilever
            'ITC.NS',         # ITC Limited
            'NESTLEIND.NS',   # Nestle India
            'BRITANNIA.NS',   # Britannia Industries
            'DABUR.NS',       # Dabur India
            'MARICO.NS',      # Marico
            'GODREJCP.NS',    # Godrej Consumer Products
            'COLPAL.NS',      # Colgate-Palmolive
            'TATACONSUM.NS',  # Tata Consumer Products
            'UBL.NS',         # United Breweries
            'VBL.NS',         # Varun Beverages
            
            # Pharmaceuticals
            'SUNPHARMA.NS',   # Sun Pharmaceutical
            'DRREDDY.NS',     # Dr. Reddy's Laboratories
            'CIPLA.NS',       # Cipla
            'DIVISLAB.NS',    # Divi's Laboratories
            'BIOCON.NS',      # Biocon
            'LUPIN.NS',       # Lupin
            'TORNTPHARM.NS',  # Torrent Pharmaceuticals
            'ALKEM.NS',       # Alkem Laboratories
            'GLAND.NS',       # Gland Pharma
            'AUROPHARMA.NS',  # Aurobindo Pharma
            
            # Healthcare Services
            'APOLLOHOSP.NS',  # Apollo Hospitals
            'FORTIS.NS',      # Fortis Healthcare
            'MAXHEALTH.NS',   # Max Healthcare
            'MEDANTA.NS',     # Global Health
            'METROPOLIS.NS',  # Metropolis Healthcare
            'LALPATHLAB.NS',  # Dr. Lal PathLabs
            
            # Construction & Infrastructure
            'LT.NS',          # Larsen & Toubro
            'ADANIENT.NS',    # Adani Enterprises
            'ADANIPORTS.NS',  # Adani Ports
            'DLF.NS',         # DLF Limited
            'GODREJPROP.NS',  # Godrej Properties
            'OBEROIRLTY.NS',  # Oberoi Realty
            'PRESTIGE.NS',    # Prestige Estates
            'PHOENIXLTD.NS',  # Phoenix Mills
            'BRIGADE.NS',     # Brigade Enterprises
            
            # Cement
            'ULTRACEMCO.NS',  # UltraTech Cement
            'SHREECEM.NS',    # Shree Cement
            'AMBUJACEM.NS',   # Ambuja Cements
            'ACC.NS',         # ACC
            'RAMCOCEM.NS',    # Ramco Cements
            'DALBHARAT.NS',   # Dalmia Bharat
            'JKCEMENT.NS',    # JK Cement
            'BIRLAMONEY.NS',  # Birla Corporation
            
            # Chemicals & Fertilizers
            'PIDILITIND.NS',  # Pidilite Industries
            'UPL.NS',         # UPL Limited
            'SRF.NS',         # SRF Limited
            'DEEPAKNITRITE.NS', # Deepak Nitrite
            'ALKYLAMINE.NS',  # Alkyl Amines
            'NAVINFLUOR.NS',  # Navin Fluorine
            'FLUOROCHEM.NS',  # Gujarat Fluorochemicals
            'ATUL.NS',        # Atul Ltd
            
            # Telecommunications
            'BHARTIARTL.NS',  # Bharti Airtel
            'IDEA.NS',        # Vodafone Idea
            'TATACOMM.NS',    # Tata Communications
            'ROUTE.NS',       # Route Mobile
            'TANLA.NS',       # Tanla Platforms
            
            # Media & Entertainment
            'ZEEL.NS',        # Zee Entertainment
            'SUNTV.NS',       # Sun TV Network
            'PVR.NS',         # PVR INOX
            'TVTODAY.NS',     # TV Today Network
            
            # Retail
            'DMART.NS',       # Avenue Supermarts
            'TRENT.NS',       # Trent
            'VMART.NS',       # V-Mart Retail
            'SHOPERSTOP.NS',  # Shoppers Stop
            'ABFRL.NS',       # Aditya Birla Fashion
            
            # Aviation & Logistics
            'INDIGO.NS',      # InterGlobe Aviation
            'CONCOR.NS',      # Container Corporation
            'BLUEDARTTEX.NS', # Blue Dart Express
            'VRL.NS',         # VRL Logistics
            'MAHLOG.NS',      # Mahindra Logistics
        ]
        
    def fetch_data(self, symbol, period='1y'):
        """Fetch historical data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            logger.info(f"Successfully fetched data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # Calculate RSI
            df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
            
            # Calculate MACD
            df['MACD'], df['Signal'], df['MACD_Hist'] = talib.MACD(
                df['Close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            
            # Calculate Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
                df['Close'],
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2
            )
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None

    def generate_signals(self, df):
        """Generate buy/sell signals based on technical indicators"""
        try:
            signals = pd.Series(['HOLD'] * len(df), index=df.index)
            
            # RSI signals
            signals[df['RSI'] < 30] = 'BUY'  # Oversold
            signals[df['RSI'] > 70] = 'SELL'  # Overbought
            
            # MACD signals
            macd_buy = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
            macd_sell = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))
            signals[macd_buy] = 'BUY'
            signals[macd_sell] = 'SELL'
            
            # Bollinger Bands signals
            signals[df['Close'] < df['BB_Lower']] = 'BUY'  # Price below lower band
            signals[df['Close'] > df['BB_Upper']] = 'SELL'  # Price above upper band
            
            df['Signal'] = signals
            return df
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return None

    def analyze_stock(self, symbol):
        """Complete analysis for a single stock"""
        try:
            # Fetch data
            df = self.fetch_data(symbol)
            if df is None:
                return None
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            if df is None:
                return None
            
            # Generate signals
            df = self.generate_signals(df)
            if df is None:
                return None
            
            return df
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {str(e)}")
            return None

    def run_analysis(self):
        """Run analysis for all configured symbols"""
        results = {}
        for symbol in self.symbols:
            logger.info(f"Analyzing {symbol}")
            df = self.analyze_stock(symbol)
            if df is not None and not df.empty:
                latest_data = df.iloc[-1]
                results[symbol] = {
                    'Close': latest_data['Close'],
                    'RSI': latest_data['RSI'],
                    'MACD': latest_data['MACD'],
                    'Signal': latest_data['Signal'],
                    'BB_Position': 'Middle'
                }
                if latest_data['Close'] > latest_data['BB_Upper']:
                    results[symbol]['BB_Position'] = 'Above Upper'
                elif latest_data['Close'] < latest_data['BB_Lower']:
                    results[symbol]['BB_Position'] = 'Below Lower'
                
                logger.info(f"{symbol}: Current Signal - {latest_data['Signal']}")
            else:
                logger.warning(f"Skipping {symbol} due to missing data")
        
        return results

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
                    
                    # Calculate volatility (standard deviation of daily returns)
                    daily_returns = df['Close'].pct_change()
                    volatility = daily_returns.std() * 100
                    sector_data['volatility'].append(volatility)
                    
                    # Calculate relative strength (current price vs 50-day MA)
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

if __name__ == "__main__":
    analyzer = IndianStockAnalyzer()
    results = analyzer.run_analysis()
    
    print("\nTechnical Analysis Results:")
    print("=" * 80)
    for symbol, data in results.items():
        print(f"\nStock: {symbol}")
        print(f"Close Price: Rs. {data['Close']:.2f}")
        print(f"RSI: {data['RSI']:.2f}")
        print(f"MACD: {data['MACD']:.2f}")
        print(f"Signal: {data['Signal']}")
        print(f"Bollinger Bands Position: {data['BB_Position']}")
        print("-" * 40)

    # Example usage of sector performance calculation
    sectors_dict = {
        'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS'],
        'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS']
    }
    sector_performance = analyzer.get_all_sectors_performance(sectors_dict)
    print("\nSector Performance Metrics:")
    print("=" * 80)
    for sector, metrics in sector_performance.items():
        print(f"\nSector: {sector}")
        print(f"Average Return: {metrics['avg_return']:.2f}%")
        print(f"Average Volatility: {metrics['avg_volatility']:.2f}%")
        print(f"Average Strength: {metrics['avg_strength']:.2f}%")
        print(f"Average Volume: {metrics['avg_volume']:.2f}")
        print(f"Best Performer: {metrics['best_performer']:.2f}%")
        print(f"Worst Performer: {metrics['worst_performer']:.2f}%")
        print(f"Stocks Analyzed: {metrics['stocks_analyzed']}")
        print("-" * 40)
