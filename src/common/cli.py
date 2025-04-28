# src/common/cli.py

import sys
from tqdm import tqdm
from typing import Optional
import json
import datetime
from typing import Dict, Any, List


def display_results(result):
    """Display trading results in a readable format
    
    Args:
        result: Trading result data (dict with metrics, summary, trades)
    """
    if not result:
        print("No results to display")
        return
        
    # Print formatted results
    print("\n" + "="*80)
    print(" "*30 + "Trading Results")
    print("="*80)
    
    if isinstance(result, dict):
        # Check for errors
        if "error" in result:
            print(f"\nError: {result['error']}")
            return
            
        # Show metrics
        if "metrics" in result:
            print("\nPerformance Metrics:")
            print("-" * 40)
            for key, value in result["metrics"].items():
                formatted_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    print(f"{formatted_key}: {value:.4f}")
                else:
                    print(f"{formatted_key}: {value}")
        
        # Show summary statistics
        if "summary" in result:
            print("\nSummary:")
            print("-" * 40)
            for key, value in result["summary"].items():
                formatted_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    print(f"{formatted_key}: {value:.4f}")
                else:
                    print(f"{formatted_key}: {value}")
        
        # Show trade statistics
        if "trades" in result:
            trades = result["trades"]
            win_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            loss_trades = sum(1 for t in trades if t.get('pnl', 0) <= 0)
            
            print(f"\nTrade Summary: {len(trades)} trades ({win_trades} profitable, {loss_trades} unprofitable)")
            
            # Show sample trades (first 5)
            if trades:
                print("\nTrade Sample:")
                print("-" * 80)
                print(f"{'Date':<20} {'Symbol':<10} {'Type':<5} {'Price':<10} {'Quantity':<10} {'PnL':<10}")
                print("-" * 80)
                
                for trade in trades[:5]:
                    date = trade.get('timestamp', 'N/A')
                    if isinstance(date, (int, float)):
                        date = datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M')
                    
                    print(f"{str(date):<20} {trade.get('symbol', 'N/A'):<10} "
                          f"{str(trade.get('action', 'N/A')):<5} {trade.get('price', 0):<10.2f} "
                          f"{trade.get('quantity', 0):<10.2f} {trade.get('pnl', 0):<10.2f}")
                
                if len(trades) > 5:
                    print(f"\n... and {len(trades) - 5} more trades")
        
        # Save detailed results to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"results_{timestamp}.json"
        
        try:
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {result_file}")
        except Exception as e:
            print(f"\nFailed to save detailed results: {e}")
    
    else:
        # For non-dictionary results, print string directly
        print(str(result))
    
    print("\n" + "="*80)


def save_results_to_file(result, filename=None):
    """Save trading results to a file
    
    Args:
        result: Trading result data
        filename: Optional custom filename, default is timestamped
        
    Returns:
        Path to saved file
    """
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        return filename
    except Exception as e:
        print(f"Failed to save results to {filename}: {e}")
        return None

