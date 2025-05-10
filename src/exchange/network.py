#!/usr/bin/env python3
import os
import sys
import socket
import urllib.request
import urllib.error
import ssl
import time
import json
from datetime import datetime

def test_internet_connection():
    print("\n=== Testing Internet Connection ===")
    hosts = ["google.com", "github.com", "amazon.com", "cloudflare.com"]
    
    for host in hosts:
        try:
            socket.create_connection((host, 80), timeout=5)
            print(f"✓ Connection to {host} successful")
        except (socket.timeout, socket.error) as e:
            print(f"✗ Connection to {host} failed: {e}")
    
def test_api_connectivity():
    print("\n=== Testing API Connectivity ===")
    
    apis = [
        {"name": "CoinGecko", "url": "https://api.coingecko.com/api/v3/ping"},
        {"name": "Binance", "url": "https://api.binance.com/api/v3/time"},
        {"name": "Alpha Vantage", "url": "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo"}
    ]
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    for api in apis:
        try:
            start = time.time()
            with urllib.request.urlopen(api["url"], timeout=10, context=ctx) as response:
                data = response.read()
                end = time.time()
                latency = (end - start) * 1000
                print(f"✓ {api['name']} API: Status {response.status} ({latency:.1f}ms)")
        except Exception as e:
            print(f"✗ {api['name']} API failed: {str(e)}")

def test_dns_resolution():
    print("\n=== Testing DNS Resolution ===")
    
    domains = ["binance.com", "coinbase.com", "kraken.com", "ftx.com"]
    
    for domain in domains:
        try:
            ip = socket.gethostbyname(domain)
            print(f"✓ {domain} resolved to {ip}")
        except socket.gaierror as e:
            print(f"✗ {domain} resolution failed: {e}")

def run_tests():
    print(f"=== Trading System Network Test ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')}")
    print(f"Hostname: {socket.gethostname()}")
    
    test_internet_connection()
    test_dns_resolution()
    test_api_connectivity()
    
    print("\n=== Test Summary ===")
    print("Network test completed")

if __name__ == "__main__":
    run_tests()
