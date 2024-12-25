import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from technical_analysis_bot import IndianStockAnalyzer
import yfinance as yf
import json
import base64
from io import BytesIO
import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="Indian Stock Market Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for download buttons
st.markdown("""
    <style>
    .stDownloadButton {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        border: none;
        cursor: pointer;
        margin: 4px;
    }
    .stDownloadButton:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

def export_to_excel(df):
    """Export DataFrame to Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write main data
        df.to_excel(writer, sheet_name='Stock Data', index=True)
        
        # Get workbook and add formats
        workbook = writer.book
        worksheet = writer.sheets['Stock Data']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#D9EAD3',
            'border': 1
        })
        
        # Apply formats
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num + 1, value, header_format)
            
        # Adjust column widths
        worksheet.set_column(0, len(df.columns), 15)
    
    return output.getvalue()

def get_table_download_link(df, filename, text):
    """Generate download link for dataframe"""
    if filename.endswith('.xlsx'):
        val = export_to_excel(df)
        b64 = base64.b64encode(val).decode()
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif filename.endswith('.csv'):
        val = df.to_csv(index=True).encode()
        b64 = base64.b64encode(val).decode()
        mime_type = 'text/csv'
    elif filename.endswith('.json'):
        val = df.to_json(orient='records', date_format='iso').encode()
        b64 = base64.b64encode(val).decode()
        mime_type = 'application/json'
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" class="stDownloadButton">{text}</a>'
    return href

def export_chart_as_html(fig, filename):
    """Export Plotly chart as HTML"""
    buffer = BytesIO()
    html_str = fig.to_html(include_plotlyjs=True, full_html=True)
    buffer.write(html_str.encode('utf-8'))
    buffer.seek(0)
    html_bytes = buffer.getvalue()
    b64 = base64.b64encode(html_bytes).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}" class="stDownloadButton">Download Interactive Chart</a>'
    return href

# Title and description
st.title("📈 Indian Stock Market Technical Analysis")
st.markdown("""
This application provides technical analysis for major Indian stocks using various indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
""")

# Initialize the analyzer
analyzer = IndianStockAnalyzer()

# Sidebar for stock selection and timeframe
st.sidebar.header("Settings")

# Group stocks by sector
sectors = {
    "Large Cap Banks & NBFCs": [s for s in analyzer.symbols if s in [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS',
        'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS',
        'AUBANK.NS', 'BANDHANBNK.NS'
    ]],
    "Financial Services & Insurance": [s for s in analyzer.symbols if s in [
        'BAJFINANCE.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIGI.NS', 'HDFCAMC.NS',
        'MUTHOOTFIN.NS', 'CHOLAFIN.NS', 'LTIM.NS', 'PFC.NS', 'RECLTD.NS'
    ]],
    "IT & Technology": [s for s in analyzer.symbols if s in [
        'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTTS.NS',
        'PERSISTENT.NS', 'MPHASIS.NS', 'COFORGE.NS', 'CYIENT.NS', 'MINDTREE.NS'
    ]],
    "Oil & Gas": [s for s in analyzer.symbols if s in [
        'RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'GAIL.NS', 'HINDPETRO.NS',
        'PETRONET.NS', 'IGL.NS', 'MGL.NS', 'GUJGASLTD.NS'
    ]],
    "Power & Energy": [s for s in analyzer.symbols if s in [
        'NTPC.NS', 'POWERGRID.NS', 'ADANIPOWER.NS', 'TATAPOWER.NS', 'TORNTPOWER.NS',
        'NHPC.NS', 'SUZLON.NS', 'ADANIGREEN.NS', 'JSL.NS'
    ]],
    "Metals & Mining": [s for s in analyzer.symbols if s in [
        'TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'SAIL.NS', 'JINDALSTEL.NS',
        'NATIONALUM.NS', 'COALINDIA.NS', 'NMDC.NS', 'VEDL.NS', 'HINDZINC.NS'
    ]],
    "Automobiles & Auto Components": [s for s in analyzer.symbols if s in [
        'TATAMOTORS.NS', 'M&M.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS',
        'ASHOKLEY.NS', 'TVSMOTOR.NS', 'HEROMOTOCO.NS', 'BALKRISIND.NS', 'MRF.NS',
        'APOLLOTYRE.NS'
    ]],
    "Consumer Goods & FMCG": [s for s in analyzer.symbols if s in [
        'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS',
        'MARICO.NS', 'GODREJCP.NS', 'COLPAL.NS', 'TATACONSUM.NS', 'UBL.NS', 'VBL.NS'
    ]],
    "Pharmaceuticals": [s for s in analyzer.symbols if s in [
        'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'BIOCON.NS',
        'LUPIN.NS', 'TORNTPHARM.NS', 'ALKEM.NS', 'GLAND.NS', 'AUROPHARMA.NS'
    ]],
    "Healthcare Services": [s for s in analyzer.symbols if s in [
        'APOLLOHOSP.NS', 'FORTIS.NS', 'MAXHEALTH.NS', 'MEDANTA.NS',
        'METROPOLIS.NS', 'LALPATHLAB.NS'
    ]],
    "Construction & Infrastructure": [s for s in analyzer.symbols if s in [
        'LT.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'DLF.NS', 'GODREJPROP.NS',
        'OBEROIRLTY.NS', 'PRESTIGE.NS', 'PHOENIXLTD.NS', 'BRIGADE.NS'
    ]],
    "Cement": [s for s in analyzer.symbols if s in [
        'ULTRACEMCO.NS', 'SHREECEM.NS', 'AMBUJACEM.NS', 'ACC.NS', 'RAMCOCEM.NS',
        'DALBHARAT.NS', 'JKCEMENT.NS', 'BIRLAMONEY.NS'
    ]],
    "Chemicals & Fertilizers": [s for s in analyzer.symbols if s in [
        'PIDILITIND.NS', 'UPL.NS', 'SRF.NS', 'DEEPAKNITRITE.NS', 'ALKYLAMINE.NS',
        'NAVINFLUOR.NS', 'FLUOROCHEM.NS', 'ATUL.NS'
    ]],
    "Telecommunications": [s for s in analyzer.symbols if s in [
        'BHARTIARTL.NS', 'IDEA.NS', 'TATACOMM.NS', 'ROUTE.NS', 'TANLA.NS'
    ]],
    "Media & Entertainment": [s for s in analyzer.symbols if s in [
        'ZEEL.NS', 'SUNTV.NS', 'PVR.NS', 'TVTODAY.NS'
    ]],
    "Retail": [s for s in analyzer.symbols if s in [
        'DMART.NS', 'TRENT.NS', 'VMART.NS', 'SHOPERSTOP.NS', 'ABFRL.NS'
    ]],
    "Aviation & Logistics": [s for s in analyzer.symbols if s in [
        'INDIGO.NS', 'CONCOR.NS', 'BLUEDARTTEX.NS', 'VRL.NS', 'MAHLOG.NS'
    ]]
}

timeframes = {
    '1 Month': '1mo',
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '2 Years': '2y',
    '5 Years': '5y'
}

# Sector selection
selected_sector = st.sidebar.selectbox("Select Sector", list(sectors.keys()))

# Stock selection
selected_stock = st.sidebar.selectbox(
    "Select Stock",
    sectors[selected_sector],
    format_func=lambda x: x.replace('.NS', '')
)

# Timeframe selection
timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

# Fetch and analyze data
df = analyzer.fetch_data(selected_stock, period=timeframe)
if df is not None and not df.empty:
    df = analyzer.calculate_indicators(df)
    df = analyzer.generate_signals(df)

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Price & Signals", "📊 Technical Indicators", "📑 Data Table", "💾 Export Data", "Sector Performance"])

    with tab1:
        # Create the main price chart with signals
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3])

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='Upper BB',
                line=dict(color='gray', dash='dash'),
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='Lower BB',
                line=dict(color='gray', dash='dash'),
                fill='tonexty'
            ),
            row=1, col=1
        )

        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{selected_stock.replace(".NS", "")} - Price Chart',
            yaxis_title='Price',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # Add download button for the chart
        st.markdown(export_chart_as_html(fig, f"{selected_stock}_chart.html"), unsafe_allow_html=True)

        # Display current signals and metrics
        latest_data = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"₹{latest_data['Close']:.2f}", 
                     f"{((latest_data['Close']/df.iloc[-2]['Close'])-1)*100:.2f}%")
        
        with col2:
            st.metric("RSI", f"{latest_data['RSI']:.2f}")
        
        with col3:
            st.metric("MACD", f"{latest_data['MACD']:.2f}")
        
        with col4:
            signal_color = {
                'BUY': 'green',
                'SELL': 'red',
                'HOLD': 'gray'
            }
            st.markdown(f"**Signal:** :{signal_color[latest_data['Signal']]}\[{latest_data['Signal']}\]")

    with tab2:
        # Technical indicators plots
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=('RSI', 'MACD'))

        # RSI
        fig2.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI'),
            row=1, col=1
        )
        fig2.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig2.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

        # MACD
        fig2.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD'),
            row=2, col=1
        )
        fig2.add_trace(
            go.Scatter(x=df.index, y=df['Signal'], name='Signal Line'),
            row=2, col=1
        )

        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Add download button for technical indicators chart
        st.markdown(export_chart_as_html(fig2, f"{selected_stock}_indicators.html"), unsafe_allow_html=True)

    with tab3:
        # Data table with latest records
        df_display = df.tail(50).sort_index(ascending=False)[
            ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'Signal']
        ].copy()
        
        # Format the numeric columns
        df_display['Open'] = df_display['Open'].apply(lambda x: f'₹{x:.2f}')
        df_display['High'] = df_display['High'].apply(lambda x: f'₹{x:.2f}')
        df_display['Low'] = df_display['Low'].apply(lambda x: f'₹{x:.2f}')
        df_display['Close'] = df_display['Close'].apply(lambda x: f'₹{x:.2f}')
        df_display['Volume'] = df_display['Volume'].apply(lambda x: f'{x:,.0f}')
        df_display['RSI'] = df_display['RSI'].apply(lambda x: f'{x:.2f}')
        df_display['MACD'] = df_display['MACD'].apply(lambda x: f'{x:.2f}')
        
        st.dataframe(df_display)
    
    with tab4:
        st.header("Export Options")
        
        # Create export section with columns
        col1, col2, col3 = st.columns(3)
        
        # Prepare data for export
        export_df = df.copy()
        
        # Format datetime index
        export_df.index = export_df.index.strftime('%Y-%m-%d %H:%M:%S')
        
        with col1:
            st.markdown("### Excel Export")
            st.markdown(get_table_download_link(
                export_df,
                f"{selected_stock}_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx",
                "Download Excel"
            ), unsafe_allow_html=True)
            
        with col2:
            st.markdown("### CSV Export")
            st.markdown(get_table_download_link(
                export_df,
                f"{selected_stock}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                "Download CSV"
            ), unsafe_allow_html=True)
            
        with col3:
            st.markdown("### JSON Export")
            st.markdown(get_table_download_link(
                export_df,
                f"{selected_stock}_{datetime.datetime.now().strftime('%Y%m%d')}.json",
                "Download JSON"
            ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Add chart export options
        st.header("Chart Export Options")
        st.markdown("""
        The interactive charts can be downloaded as HTML files, which can be opened in any web browser.
        These files contain all the interactive features and can be used offline.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Price Chart")
            st.markdown(export_chart_as_html(fig, f"{selected_stock}_price_chart.html"), unsafe_allow_html=True)
            
        with col2:
            st.markdown("### Technical Indicators")
            st.markdown(export_chart_as_html(fig2, f"{selected_stock}_indicators_chart.html"), unsafe_allow_html=True)

    with tab5:
        st.header("Sector Performance Analysis")
        
        # Add interactive filters in sidebar
        st.sidebar.markdown("### Sector Analysis Filters")
        
        # Time period selection for sector analysis
        selected_period = st.sidebar.selectbox(
            "Select Time Period",
            list(timeframes.keys()),
            index=3  # Default to 1 Year
        )
        
        # Minimum number of stocks filter
        min_stocks = st.sidebar.slider(
            "Minimum Stocks per Sector",
            min_value=1,
            max_value=20,
            value=3,
            help="Filter sectors with at least this many stocks"
        )
        
        # Performance threshold filter
        perf_threshold = st.sidebar.slider(
            "Performance Threshold (%)",
            min_value=-50,
            max_value=50,
            value=-10,
            help="Filter sectors with returns above this threshold"
        )
        
        # Sort options
        sort_metric = st.sidebar.selectbox(
            "Sort Sectors By",
            ["Average Return", "Volatility", "Relative Strength", "Number of Stocks"],
            index=0
        )
        
        sort_order = st.sidebar.radio(
            "Sort Order",
            ["Descending", "Ascending"],
            horizontal=True
        )
        
        # Calculate sector performance
        sectors_performance = analyzer.get_all_sectors_performance(sectors, timeframes[selected_period])
        
        if sectors_performance:
            # Filter and sort sectors based on user preferences
            filtered_sectors = {
                k: v for k, v in sectors_performance.items() 
                if v['stocks_analyzed'] >= min_stocks and v['avg_return'] >= perf_threshold
            }
            
            # Prepare data for plots
            sector_names = list(filtered_sectors.keys())
            
            if not sector_names:
                st.warning("No sectors match the selected filters. Try adjusting the criteria.")
            else:
                # Add performance summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    best_sector = max(filtered_sectors.items(), key=lambda x: x[1]['avg_return'])
                    st.metric(
                        "Best Performing Sector",
                        best_sector[0],
                        f"{best_sector[1]['avg_return']:.1f}%"
                    )
                
                with col2:
                    avg_market_return = np.mean([s['avg_return'] for s in filtered_sectors.values()])
                    st.metric(
                        "Average Market Return",
                        f"{avg_market_return:.1f}%"
                    )
                
                with col3:
                    lowest_vol_sector = min(filtered_sectors.items(), key=lambda x: x[1]['avg_volatility'])
                    st.metric(
                        "Most Stable Sector",
                        lowest_vol_sector[0],
                        f"{lowest_vol_sector[1]['avg_volatility']:.1f}% Vol"
                    )
                
                with col4:
                    strongest_sector = max(filtered_sectors.items(), key=lambda x: x[1]['avg_strength'])
                    st.metric(
                        "Strongest Momentum",
                        strongest_sector[0],
                        f"{strongest_sector[1]['avg_strength']:.1f}% RS"
                    )
                
                # Interactive chart options
                chart_type = st.radio(
                    "Chart Type",
                    ["Bar", "Line", "Scatter"],
                    horizontal=True
                )
                
                # Returns comparison with selected chart type
                fig_returns = go.Figure()
                
                if chart_type == "Bar":
                    fig_returns.add_trace(go.Bar(
                        x=sector_names,
                        y=[metrics['avg_return'] for metrics in filtered_sectors.values()],
                        marker_color=['red' if x < 0 else 'green' for x in [metrics['avg_return'] for metrics in filtered_sectors.values()]],
                        name='Average Return'
                    ))
                elif chart_type == "Line":
                    fig_returns.add_trace(go.Scatter(
                        x=sector_names,
                        y=[metrics['avg_return'] for metrics in filtered_sectors.values()],
                        mode='lines+markers',
                        name='Average Return',
                        line=dict(color='blue')
                    ))
                else:  # Scatter
                    fig_returns.add_trace(go.Scatter(
                        x=[metrics['avg_volatility'] for metrics in filtered_sectors.values()],
                        y=[metrics['avg_return'] for metrics in filtered_sectors.values()],
                        mode='markers+text',
                        text=sector_names,
                        textposition="top center",
                        name='Risk vs Return',
                        marker=dict(
                            size=[metrics['stocks_analyzed'] * 5 for metrics in filtered_sectors.values()],
                            color=[metrics['avg_strength'] for metrics in filtered_sectors.values()],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Relative Strength")
                        )
                    ))
                
                if chart_type == "Scatter":
                    fig_returns.update_layout(
                        title='Risk vs Return Analysis',
                        xaxis_title="Volatility (%)",
                        yaxis_title="Return (%)",
                        height=600
                    )
                else:
                    fig_returns.update_layout(
                        title=f'Sector Returns Comparison ({selected_period})',
                        xaxis_title="Sectors",
                        yaxis_title="Average Return (%)",
                        height=500
                    )
                    fig_returns.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig_returns, use_container_width=True)
                
                # Add correlation heatmap
                if st.checkbox("Show Sector Correlation Analysis"):
                    st.subheader("Sector Correlation Analysis")
                    
                    # Create correlation matrix
                    corr_data = {}
                    for sector in filtered_sectors:
                        symbol = sectors[sector][0]  # Take first stock as representative
                        df = analyzer.fetch_data(symbol, timeframes[selected_period])
                        if df is not None:
                            corr_data[sector] = df['Close'].pct_change()
                    
                    corr_df = pd.DataFrame(corr_data).corr()
                    
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_df.values,
                        x=corr_df.columns,
                        y=corr_df.index,
                        colorscale='RdBu',
                        zmin=-1,
                        zmax=1
                    ))
                    
                    fig_corr.update_layout(
                        title='Sector Correlation Heatmap',
                        height=600
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Add performance distribution
                if st.checkbox("Show Performance Distribution"):
                    st.subheader("Performance Distribution Analysis")
                    
                    fig_dist = go.Figure()
                    
                    for sector in filtered_sectors:
                        returns = []
                        for symbol in sectors[sector]:
                            df = analyzer.fetch_data(symbol, timeframes[selected_period])
                            if df is not None:
                                ret = ((df['Close'][-1] - df['Close'][0]) / df['Close'][0]) * 100
                                returns.append(ret)
                        
                        fig_dist.add_trace(go.Box(
                            y=returns,
                            name=sector,
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-1.8
                        ))
                    
                    fig_dist.update_layout(
                        title='Sector Performance Distribution',
                        yaxis_title='Return (%)',
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Detailed metrics table with sorting
                st.header("Detailed Sector Metrics")
                metrics_df = pd.DataFrame({
                    'Sector': sector_names,
                    'Avg Return (%)': [f"{metrics['avg_return']:.2f}" for metrics in filtered_sectors.values()],
                    'Volatility (%)': [f"{metrics['avg_volatility']:.2f}" for metrics in filtered_sectors.values()],
                    'Relative Strength (%)': [f"{metrics['avg_strength']:.2f}" for metrics in filtered_sectors.values()],
                    'Best Performer (%)': [f"{metrics['best_performer']:.2f}" for metrics in filtered_sectors.values()],
                    'Worst Performer (%)': [f"{metrics['worst_performer']:.2f}" for metrics in filtered_sectors.values()],
                    'Stocks Analyzed': [metrics['stocks_analyzed'] for metrics in filtered_sectors.values()]
                })
                
                # Sort based on user selection
                sort_col = {
                    "Average Return": "Avg Return (%)",
                    "Volatility": "Volatility (%)",
                    "Relative Strength": "Relative Strength (%)",
                    "Number of Stocks": "Stocks Analyzed"
                }[sort_metric]
                
                metrics_df = metrics_df.sort_values(
                    sort_col,
                    ascending=(sort_order == "Ascending"),
                    key=lambda x: pd.to_numeric(x, errors='coerce')
                )
                
                # Display table with formatting
                st.dataframe(
                    metrics_df.style.background_gradient(subset=['Avg Return (%)'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # Download buttons for data export
                col1, col2 = st.columns(2)
                with col1:
                    csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        "Download Sector Analysis (CSV)",
                        csv,
                        "sector_analysis.csv",
                        "text/csv",
                        key='download-csv'
                    )
                
                with col2:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        metrics_df.to_excel(writer, sheet_name='Sector Analysis', index=False)
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        "Download Sector Analysis (Excel)",
                        excel_data,
                        "sector_analysis.xlsx",
                        "application/vnd.ms-excel",
                        key='download-excel'
                    )

else:
    st.error("Unable to fetch data for the selected stock. Please try another stock or timeframe.")
