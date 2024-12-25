# Indian Stock Market Technical Analysis Bot

This Python script implements a technical analysis bot specifically designed for the Indian stock market. It uses Yahoo Finance for data fetching and TA-Lib for calculating various technical indicators.

## Features

- Fetches real-time and historical data for Indian stocks using Yahoo Finance
- Calculates key technical indicators:
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
- Generates buy/sell signals based on indicator combinations
- Includes error handling and logging
- Focuses on top Indian stocks (configurable)

## Prerequisites

- Python 3.8 or higher
- TA-Lib installation (see installation instructions below)

## Installation

1. Clone this repository or download the files
2. Install TA-Lib:
   - Windows: Download and install the appropriate wheel file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   - Linux: `sudo apt-get install ta-lib`
   - macOS: `brew install ta-lib`

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script:
```bash
python technical_analysis_bot.py
```

The script will analyze the configured stocks and output:
- Current stock price
- RSI value
- MACD indicators
- Current signal (BUY/SELL/HOLD)
- Bollinger Bands position

## Customization

You can modify the `symbols` list in the `IndianStockAnalyzer` class to analyze different stocks. Make sure to use the correct Yahoo Finance symbols (usually with .NS suffix for NSE-listed stocks).

## Note

This bot is for educational purposes only. Always do your own research and consider multiple factors before making investment decisions.
