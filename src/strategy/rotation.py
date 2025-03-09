import pandas as pd
import ccxt
import os
from dotenv import load_dotenv

# Load environment variables from .env file (recommended for API keys)
load_dotenv()

pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

# Get API credentials from environment variables instead of hardcoding
exchange_params = {
    'apiKey': 'HbMs2aPHQfPNh097P8SGWl7PpcjyUUk6I4WURcn6SUaTWjJ8SM8jb4tYSZgLFoBX',
    'secret': 'vVQe126DGUw9BNwSG4gkt9E2DTz1YtiGnP3KzmVWH8AFZJ1SAEsPdOexbuwT68ck',
    'timeout': 60000,  # Increased timeout from 30s to 60s
    'enableRateLimit': True,
    'options': {
        'adjustForTimeDifference': True,
        'recvWindow': 60000,
        'defaultType': 'spot'  # Specify the market type explicitly
    },
    # Use proxies if needed for bypassing network restrictions
    'proxies': {
        'http': "http://127.0.0.1:7890",       # HTTP 代理地址
        'https': "http://127.0.0.1:7890"      # HTTPS 代理地址
    }
}

def fetch_crypto_data(exchange, symbol, time_interval, n_days):
    """Fetch and process OHLCV data with error handling"""
    try:
        data = exchange.fetch_ohlcv(symbol=symbol, timeframe=time_interval, since=1701388800000, limit=n_days + 10)
        
        # If we still don't have data after all retries
        if 'data' not in locals() or not data:
            print(f"Could not fetch data for {symbol} after multiple attempts")
            return None
        
        # Handle empty data case
        if not data or len(data) < n_days:
            print(f"Warning: Insufficient data for {symbol}")
            return None
            
        # Process data
        df = pd.DataFrame(data, dtype=float)
        df.rename(columns={0: 'MTS', 1: 'open', 2: 'high',
                           3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
        df['candle_begin_time'] = pd.to_datetime(df['MTS'], unit='ms')
        df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]
        
        # Calculate N-day change - ensure we have enough data first
        if len(df) >= n_days + 1:  # Need at least N+1 days for N-day change
            df[f'{n_days}天涨跌幅'] = df['close'].pct_change(n_days)
            return df
        else:
            print(f"Warning: Not enough data for {symbol} to calculate {n_days}-day change")
            return None
            
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def main():
    try:
        # Create exchange connection
        exchange = ccxt.binance(exchange_params)
        
        # Set parameters
        time_interval = '1d'
        N = 20  # Calculate N-day change
        symbols = ['BTC/USDT', 'ETH/USDT']
        
        # Fetch and process data for each symbol
        change_dict = {}
        dataframes = {}
        
        for symbol in symbols:
            data = exchange.fetch_ohlcv(symbol=symbol, timeframe=time_interval, since=1701388800000, limit= 10)

            df = fetch_crypto_data(exchange, symbol, time_interval, N)
            if df is not None:
                dataframes[symbol] = df
                # Get the most recent N-day change value
                change_dict[symbol] = df.iloc[-1][f'{N}天涨跌幅']
                print(f"{symbol} {N}天涨跌幅: {change_dict[symbol]:.2%}")
            else:
                print(f"Skipping {symbol} due to data issues")
        
        # Trading decision logic - fixed to properly handle all cases
        if len(change_dict) == 2:  # Ensure we have data for both symbols
            btc_change = change_dict.get('BTC/USDT')
            eth_change = change_dict.get('ETH/USDT')
            
            # Case 1: Both negative
            if btc_change < 0 and eth_change < 0:
                print('比特币和以太坊涨幅都<0，空仓')
            # Case 2: BTC better than ETH
            elif btc_change > eth_change:
                if btc_change > 0:
                    print('比特币涨幅大于0且大于以太坊涨幅，买入比特币')
                else:
                    print('比特币跌幅小于以太坊跌幅，买入比特币')
            # Case 3: ETH better than BTC
            elif eth_change > btc_change:
                if eth_change > 0:
                    print('以太坊涨幅大于0且大于比特币涨幅，买入以太坊')
                else:
                    print('以太坊跌幅小于比特币跌幅，买入以太坊')
            # Case 4: Equal (unlikely but possible)
            else:
                print('比特币和以太坊涨幅相同，建议平分资金')
        else:
            print("无法做出交易决策，数据不完整")
            
    except Exception as e:
        print(f"程序运行出错: {e}")

# Add option to use cached data if API fails
def use_cached_data(symbols, n_days):
    """Fallback function to use cached data if API calls fail"""
    import os
    import datetime
    
    print("Attempting to use cached data...")
    result = {}
    
    for symbol in symbols:
        symbol_file = symbol.replace('/', '_') + '.csv'
        if os.path.exists(symbol_file):
            try:
                # Load cached data
                df = pd.read_csv(symbol_file)
                df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
                
                # Check if data is recent enough (within 1 day)
                latest_date = df['candle_begin_time'].max()
                today = pd.Timestamp(datetime.datetime.now())
                
                if (today - latest_date).days <= 1:
                    # Calculate N-day change
                    if len(df) >= n_days + 1:
                        if f'{n_days}天涨跌幅' not in df.columns:
                            df[f'{n_days}天涨跌幅'] = df['close'].pct_change(n_days)
                        
                        result[symbol] = df
                        print(f"Using cached data for {symbol}, last update: {latest_date}")
                    else:
                        print(f"Cached data for {symbol} does not have enough history")
                else:
                    print(f"Cached data for {symbol} is too old: {latest_date}")
            except Exception as e:
                print(f"Error loading cached data for {symbol}: {e}")
    
    return result

# Modified main function to include handling network restrictions
def handle_network_restrictions():
    """Function to handle potential network restrictions"""
    print("Checking if network restrictions might be causing timeouts...")
    
    # Test ping to Binance
    import subprocess
    import platform
    
    host = "api.binance.com"
    
    # Check if we can reach Binance
    if platform.system().lower() == "windows":
        ping_cmd = ["ping", "-n", "1", host]
    else:
        ping_cmd = ["ping", "-c", "1", host]
    
    try:
        result = subprocess.run(ping_cmd, capture_output=True, text=True)
        if "time=" in result.stdout or "TTL=" in result.stdout:
            print(f"Can reach {host} - network is working")
            return False
        else:
            print(f"Cannot ping {host} - possible network restriction")
            return True
    except Exception:
        print("Could not test network connectivity")
        return True

if __name__ == "__main__":
    network_restricted = handle_network_restrictions()
    
    if network_restricted:
        print("\nNetwork appears to be restricted. Consider these options:")
        print("1. Use a VPN to bypass network restrictions")
        print("2. Configure a proxy in the exchange_params")
        print("3. Use an alternative API endpoint if available")
        print("4. Contact your network administrator\n")
    
    # Run the main program
    main()