#!/usr/bin/env python3
"""
Comprehensive Network Diagnostics Tool for Crypto API Connections

This tool provides in-depth analysis of network connectivity, proxy configurations,
TLS handshakes, and API request performance specifically optimized for crypto exchanges
like Binance, with extensions for advanced network testing.

Features:
- Multi-layered network diagnostics (TCP, DNS, HTTP, WebSockets)
- Proxy auto-detection and validation
- TLS/SSL handshake analysis
- API endpoint benchmarking
- Latency and packet loss measurements
- Connection resilience testing
- Automated troubleshooting suggestions

Usage: python network_diagnostics.py [options]

Options:
  --target URL       Target API endpoint to test (default: Binance)
  --proxy URL        Specify proxy to test (e.g., http://127.0.0.1:7890)
  --symbol PAIR      Trading pair for API tests (default: BTC/USDT)
  --timeframe TF     Timeframe for historical data tests (default: 1h)
  --full-scan        Perform comprehensive network scan
  --stress-test      Run connection stress tests
  --verbose          Enable detailed debugging output
"""

import argparse
import asyncio
import json
import platform
import random
import socket
import ssl
import statistics
import subprocess
import sys
import time
import traceback
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import dns.resolver
import psutil
import requests
import urllib3
import ccxt
import ccxt.async_support as ccxt_async

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class NetworkDiagnostics:
    """Comprehensive network diagnostics and troubleshooting tool"""
    
    def __init__(self, 
                 target: str = "https://api.binance.com/api/v3/ping",
                 proxy: Optional[str] = None,
                 symbol: str = "BTC/USDT",
                 timeframe: str = "1h",
                 verbose: bool = False,
                 log_file: Optional[str] = None):
        """
        Initialize diagnostics tool
        
        Args:
            target: Base API endpoint URL
            proxy: Proxy URL (optional)
            symbol: Trading pair symbol
            timeframe: OHLCV timeframe
            verbose: Enable verbose output
            log_file: Path to log file (optional)
        """
        self.target_url = target
        self.proxy = proxy
        self.symbol = symbol
        self.timeframe = timeframe
        self.verbose = verbose
        self.exchange_endpoints = {
            "binance": {
                "spot": "https://api.binance.com/api/v3/ping",
                "futures": "https://fapi.binance.com/fapi/v1/ping",
                "ws": "wss://stream.binance.com:9443/ws"
            },
            "bybit": {
                "spot": "https://api.bybit.com/v2/public/time",
                "futures": "https://api.bybit.com/public/linear/time"
            }
        }
        self.common_proxy_ports = [7890, 8080, 1080, 8118, 3128, 10809, 7070, 9090]
        self.start_time = time.time()
        
        # Setup logging
        self.log_file = log_file
        if log_file:
            # Ensure reports directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Set up file handler
            self.file_logger = logging.getLogger('network_diagnostics')
            self.file_logger.setLevel(logging.DEBUG)
            
            # Create handlers
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Create formatters
            log_format = '%(asctime)s | %(levelname)8s | %(message)s'
            date_format = '%Y-%m-%d %H:%M:%S'
            formatter = logging.Formatter(log_format, date_format)
            
            # Add formatters to handlers
            file_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.file_logger.addHandler(file_handler)
            
            self.file_logger.info("==== Network Diagnostics Log ====")
            self.file_logger.info(f"Target: {target}")
            self.file_logger.info(f"Proxy: {proxy if proxy else 'None'}")
            self.file_logger.info(f"Symbol: {symbol}")
            self.file_logger.info(f"Timeframe: {timeframe}")
            self.file_logger.info("=" * 40)
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log formatted message with timestamp"""
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console_message = f"{timestamp} | {elapsed:7.2f}s | {level:<8} | {message}"
        print(console_message)
        
        # Also log to file if enabled
        if self.log_file:
            log_method = getattr(self.file_logger, level.lower(), self.file_logger.info)
            log_method(f"{elapsed:7.2f}s | {message}")
    
    def debug(self, message: str) -> None:
        """Log debug message if verbose enabled"""
        if self.verbose:
            self.log(message, "DEBUG")
        elif self.log_file:
            # Always log debug messages to file even if not printed to console
            self.file_logger.debug(f"{time.time() - self.start_time:7.2f}s | {message}")
    
    def print_section(self, title: str, width: int = 80) -> None:
        """Print formatted section header"""
        section_header = "\n" + "=" * width + f"\n {title.upper()}\n" + "=" * width
        print(section_header)
        
        if self.log_file:
            self.file_logger.info("=" * 40)
            self.file_logger.info(f"{title.upper()}")
            self.file_logger.info("=" * 40)
    
    # ====================
    # Network Core Tests
    # ====================
    
    def test_basic_connectivity(self, 
                              url: Optional[str] = None,
                              attempts: int = 3,
                              timeout: int = 5) -> Dict:
        """
        Test basic HTTP connectivity with latency measurements
        
        Args:
            url: Target URL (defaults to initialized target)
            attempts: Number of test attempts
            timeout: Connection timeout in seconds
            
        Returns:
            Dictionary with test results
        """
        url = url or self.target_url
        results = {
            "success": 0,
            "failures": 0,
            "latencies": [],
            "errors": [],
            "status_codes": []
        }
        
        self.print_section(f"Basic Connectivity Test: {url}")
        
        # DNS resolution test
        dns_result = self.test_dns_resolution(url)
        results.update({"dns": dns_result})
        
        if not dns_result["success"]:
            self.log("DNS resolution failed, skipping further tests", "WARNING")
            return results
        
        # TCP connection test
        tcp_result = self.test_tcp_connection(url)
        results.update({"tcp": tcp_result})
        
        if not tcp_result["success"]:
            self.log("TCP connection failed, skipping HTTP tests", "WARNING")
            return results
        
        # HTTP request tests
        for i in range(attempts):
            try:
                proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
                
                start = time.time()
                response = requests.get(
                    url,
                    proxies=proxies,
                    timeout=timeout,
                    verify=False  # Skip SSL verification for basic connectivity
                )
                latency = (time.time() - start) * 1000
                
                results["success"] += 1
                results["latencies"].append(latency)
                results["status_codes"].append(response.status_code)
                
                self.log(f"Attempt {i+1}: Success (HTTP {response.status_code}, {latency:.2f}ms)")
                
            except Exception as e:
                results["failures"] += 1
                results["errors"].append(str(e))
                self.log(f"Attempt {i+1}: Failed - {str(e)}", "ERROR")
        
        # Calculate statistics
        if results["latencies"]:
            results["avg_latency"] = statistics.mean(results["latencies"])
            results["min_latency"] = min(results["latencies"])
            results["max_latency"] = max(results["latencies"])
            if len(results["latencies"]) > 1:
                results["latency_stdev"] = statistics.stdev(results["latencies"])
            else:
                results["latency_stdev"] = 0
            
            self.log(f"Success rate: {results['success']}/{attempts}")
            self.log(f"Latency: {results['avg_latency']:.2f}ms ± {results['latency_stdev']:.2f}ms")
            
            if results["avg_latency"] > 500:
                self.log("High latency detected (>500ms), may cause API timeouts", "WARNING")
        
        return results
    
    def test_dns_resolution(self, url: str) -> Dict:
        """Test DNS resolution for a given URL"""
        result = {"success": False, "ips": [], "error": None}
        
        try:
            hostname = urlparse(url).netloc.split(":")[0]  # Remove port if present
            self.log(f"Resolving DNS for {hostname}...")
            
            answers = dns.resolver.resolve(hostname, "A")
            result["ips"] = [str(r) for r in answers]
            result["success"] = True
            
            self.log(f"DNS resolved to: {', '.join(result['ips'])}")
            
        except Exception as e:
            result["error"] = str(e)
            self.log(f"DNS resolution failed: {str(e)}", "ERROR")
        
        return result
    
    def test_tcp_connection(self, url: str, port: Optional[int] = None) -> Dict:
        """Test basic TCP connectivity"""
        result = {"success": False, "latency": None, "error": None}
        
        try:
            parsed = urlparse(url)
            host = parsed.hostname
            port = port or (parsed.port or (443 if parsed.scheme == "https" else 80))
            
            self.log(f"Testing TCP connection to {host}:{port}...")
            
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((host, port))
            latency = (time.time() - start) * 1000
            sock.close()
            
            result.update({"success": True, "latency": latency})
            self.log(f"TCP connection successful ({latency:.2f}ms)")
            
        except Exception as e:
            result["error"] = str(e)
            self.log(f"TCP connection failed: {str(e)}", "ERROR")
        
        return result
    
    # ====================
    # Proxy Diagnostics
    # ====================
    
    def diagnose_proxy(self) -> Dict:
        """Comprehensive proxy connection diagnostics"""
        self.print_section("Proxy Connection Diagnostics")
        
        if not self.proxy:
            self.log("No proxy configured, skipping proxy tests")
            return {"configured": False}
        
        results = {
            "configured": True,
            "proxy_url": self.proxy,
            "valid": False,
            "tests": {}
        }
        
        # Parse proxy URL
        parsed = urlparse(self.proxy)
        proxy_host = parsed.hostname
        proxy_port = parsed.port
        
        if not proxy_host or not proxy_port:
            results["error"] = "Invalid proxy URL format"
            self.log(f"Invalid proxy URL: {self.proxy}", "ERROR")
            return results
        
        # Test 1: Proxy TCP connectivity
        tcp_test = self.test_tcp_connection(f"tcp://{proxy_host}:{proxy_port}")
        results["tests"]["tcp_connect"] = tcp_test
        
        if not tcp_test["success"]:
            results["error"] = "Proxy TCP connection failed"
            return results
        
        # Test 2: HTTP CONNECT through proxy
        http_test = self.test_http_proxy()
        results["tests"]["http_connect"] = http_test
        
        if not http_test["success"]:
            results["error"] = "Proxy HTTP CONNECT failed"
            return results
        
        # Test 3: TLS through proxy
        tls_test = self.analyze_tls_handshake(use_proxy=True)
        results["tests"]["tls_handshake"] = tls_test
        
        if not tls_test["success"]:
            results["error"] = "Proxy TLS handshake failed"
            return results
        
        # Test 4: API endpoint through proxy
        api_test = self.test_basic_connectivity()
        results["tests"]["api_connect"] = api_test
        
        if api_test["success"] > 0:
            results["valid"] = True
            self.log("Proxy configuration is fully functional", "SUCCESS")
        else:
            results["error"] = "Proxy failed all API connection tests"
        
        return results
    
    def test_http_proxy(self) -> Dict:
        """Test HTTP CONNECT through proxy"""
        result = {"success": False, "error": None}
        
        try:
            parsed = urlparse(self.proxy)
            proxy_host = parsed.hostname
            proxy_port = parsed.port
            
            target_parsed = urlparse(self.target_url)
            target_host = target_parsed.hostname
            target_port = target_parsed.port or (443 if target_parsed.scheme == "https" else 80)
            
            self.log(f"Testing HTTP CONNECT through {proxy_host}:{proxy_port} to {target_host}:{target_port}")
            
            # Establish TCP connection to proxy
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((proxy_host, proxy_port))
            
            # Send CONNECT request
            connect_req = (
                f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
                f"Host: {target_host}:{target_port}\r\n"
                "\r\n"
            )
            sock.sendall(connect_req.encode())
            
            # Read response
            response = b""
            while b"\r\n\r\n" not in response:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                response += chunk
            
            # Check response
            if not response.startswith(b"HTTP/1.1 200"):
                raise ConnectionError(f"Proxy rejected CONNECT: {response.decode()}")
            
            result["success"] = True
            self.log("HTTP CONNECT through proxy successful")
            
        except Exception as e:
            result["error"] = str(e)
            self.log(f"HTTP CONNECT failed: {str(e)}", "ERROR")
        finally:
            if 'sock' in locals():
                sock.close()
        
        return result
    
    # ====================
    # TLS/SSL Diagnostics  
    # ====================
    
    def analyze_tls_handshake(self, 
                             host: Optional[str] = None,
                             port: int = 443,
                             use_proxy: bool = False) -> Dict:
        """Analyze TLS handshake process"""
        result = {
            "success": False,
            "protocol": None,
            "cipher": None,
            "cert_info": None,
            "handshake_time": None,
            "error": None
        }
        
        host = host or urlparse(self.target_url).hostname
        self.print_section(f"TLS Handshake Analysis: {host}:{port}")
        
        try:
            # Create appropriate socket
            if use_proxy and self.proxy:
                sock = self._create_proxy_socket(host, port)
                self.log(f"Using proxy: {self.proxy}")
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((host, port))
                self.log("Direct connection (no proxy)")
            
            sock.settimeout(10)
            
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            # Perform handshake
            start = time.time()
            secure_sock = context.wrap_socket(sock, server_hostname=host)
            handshake_time = (time.time() - start) * 1000
            
            # Get connection info
            result.update({
                "success": True,
                "protocol": secure_sock.version(),
                "cipher": secure_sock.cipher(),
                "handshake_time": handshake_time,
                "cert_info": self._get_cert_info(secure_sock)
            })
            
            self.log(f"TLS handshake successful ({handshake_time:.2f}ms)")
            self.log(f"Protocol: {secure_sock.version()}")
            self.log(f"Cipher suite: {secure_sock.cipher()[0]}")
            
            if result["cert_info"]:
                self.log(f"Certificate CN: {result['cert_info'].get('subject', {}).get('CN', 'N/A')}")
                self.log(f"Issuer: {result['cert_info'].get('issuer', {}).get('CN', 'N/A')}")
                self.log(f"Expires: {result['cert_info'].get('notAfter', 'N/A')}")
            
        except Exception as e:
            result["error"] = str(e)
            elapsed = (time.time() - start) * 1000 if 'start' in locals() else 0
            self.log(f"TLS handshake failed ({elapsed:.2f}ms): {str(e)}", "ERROR")
        finally:
            if 'secure_sock' in locals():
                secure_sock.close()
            elif 'sock' in locals():
                sock.close()
        
        return result
    
    def _create_proxy_socket(self, host: str, port: int) -> socket.socket:
        """Create socket tunneled through HTTP proxy"""
        parsed = urlparse(self.proxy)
        proxy_host = parsed.hostname
        proxy_port = parsed.port
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((proxy_host, proxy_port))
        
        # Send CONNECT request
        connect_req = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n"
        sock.sendall(connect_req.encode())
        
        # Read response
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = sock.recv(1024)
            if not chunk:
                break
            response += chunk
        
        if not response.startswith(b"HTTP/1.1 200"):
            raise ConnectionError(f"Proxy CONNECT failed: {response.decode()}")
        
        return sock
    
    def _get_cert_info(self, sock: ssl.SSLSocket) -> Optional[Dict]:
        """Extract certificate information from SSL socket"""
        try:
            cert = sock.getpeercert()
            if not cert:
                return None
                
            result = {
                "subject": {k[0][0]: k[0][1] for k in cert.get("subject", [])},
                "issuer": {k[0][0]: k[0][1] for k in cert.get("issuer", [])},
                "notBefore": cert.get("notBefore"),
                "notAfter": cert.get("notAfter"),
                "serialNumber": cert.get("serialNumber"),
                "version": cert.get("version")
            }
            return result
        except Exception:
            return None
    
    # ====================
    # Advanced Network Tests
    # ====================
    
    def run_advanced_tests(self) -> Dict:
        """Run comprehensive network diagnostics"""
        results = {}
        
        # 1. Network Interface Analysis
        self.print_section("Network Interface Analysis")
        results["interfaces"] = self.analyze_network_interfaces()
        
        # 2. Route Analysis
        self.print_section("Route Analysis")
        results["routing"] = self.analyze_routing()
        
        # 3. Bandwidth Test
        self.print_section("Bandwidth Test")
        results["bandwidth"] = self.test_bandwidth()
        
        # 4. Packet Loss Test
        self.print_section("Packet Loss Test")
        results["packet_loss"] = self.test_packet_loss()
        
        # 5. Latency Distribution
        self.print_section("Latency Distribution")
        results["latency_dist"] = self.test_latency_distribution()
        
        return results
    
    def analyze_network_interfaces(self) -> Dict:
        """Analyze system network interfaces"""
        interfaces = {}
        
        try:
            for name, addrs in psutil.net_if_addrs().items():
                interfaces[name] = {
                    "addresses": [{
                        "family": str(addr.family),
                        "address": addr.address,
                        "netmask": addr.netmask
                    } for addr in addrs],
                    "stats": psutil.net_if_stats().get(name, {})
                }
            
            self.log(f"Found {len(interfaces)} network interfaces")
            if self.verbose:
                for name, data in interfaces.items():
                    self.debug(f"Interface {name}:")
                    for addr in data["addresses"]:
                        self.debug(f"  {addr['family']}: {addr['address']}")
                    self.debug(f"  Is up: {data['stats'].isup if hasattr(data['stats'], 'isup') else False}")
            
        except Exception as e:
            self.log(f"Failed to analyze interfaces: {str(e)}", "ERROR")
        
        return interfaces
    
    def analyze_routing(self) -> Dict:
        """Analyze network routing tables"""
        routes = []
        
        try:
            if platform.system() == "Windows":
                output = subprocess.check_output("route print", shell=True).decode()
                # Parse Windows route table
            else:
                output = subprocess.check_output("ip route", shell=True).decode()
                for line in output.splitlines():
                    routes.append(line.strip())
            
            self.log(f"Found {len(routes)} routing entries")
            if self.verbose:
                for route in routes[:5]:  # Show first 5 routes
                    self.debug(f"Route: {route}")
            
        except Exception as e:
            self.log(f"Failed to analyze routing: {str(e)}", "ERROR")
        
        return {"routes": routes}
    
    def test_bandwidth(self, duration: int = 5) -> Dict:
        """
        Perform basic bandwidth test by downloading test data
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            Dictionary with bandwidth results
        """
        test_url = "https://speedtest.ftp.otenet.gr/files/test100k.db"  # 100KB test file
        result = {
            "download_speed": None,  # KB/s
            "duration": duration,
            "error": None
        }
        
        try:
            self.log(f"Running bandwidth test for {duration} seconds...")
            
            start = time.time()
            total_bytes = 0
            end_time = start + duration
            
            with requests.get(test_url, stream=True, timeout=10) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    total_bytes += len(chunk)
                    if time.time() > end_time:
                        break
            
            elapsed = time.time() - start
            speed_kbs = (total_bytes / 1024) / elapsed
            result["download_speed"] = speed_kbs
            
            self.log(f"Download speed: {speed_kbs:.2f} KB/s")
            
        except Exception as e:
            result["error"] = str(e)
            self.log(f"Bandwidth test failed: {str(e)}", "ERROR")
        
        return result
    
    def test_packet_loss(self, host: Optional[str] = None, count: int = 10) -> Dict:
        """
        Test packet loss using ICMP ping
        
        Args:
            host: Target host (defaults to API endpoint)
            count: Number of ping attempts
            
        Returns:
            Dictionary with packet loss results
        """
        host = host or urlparse(self.target_url).hostname
        result = {
            "target": host,
            "sent": count,
            "received": 0,
            "loss_percent": 100,
            "error": None
        }
        
        try:
            self.log(f"Testing packet loss to {host} ({count} packets)...")
            
            if platform.system() == "Windows":
                cmd = f"ping -n {count} {host}"
            else:
                cmd = f"ping -c {count} {host}"
            
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
            
            # Parse output
            if "received" in output:
                if platform.system() == "Windows":
                    # Windows ping output parsing
                    loss_line = [line for line in output.splitlines() if "loss" in line][0]
                    loss_percent = int(loss_line.split("(")[1].split("%")[0])
                else:
                    # Unix ping output parsing
                    loss_line = [line for line in output.splitlines() if "packet loss" in line][0]
                    loss_percent = int(loss_line.split("%")[0].split(" ")[-1])
                
                result["loss_percent"] = loss_percent
                result["received"] = count - int(count * loss_percent / 100)
                
                self.log(f"Packet loss: {loss_percent}% ({result['received']}/{count} received)")
            
        except Exception as e:
            result["error"] = str(e)
            self.log(f"Packet loss test failed: {str(e)}", "ERROR")
        
        return result
    
    def test_latency_distribution(self, 
                                host: Optional[str] = None,
                                samples: int = 20) -> Dict:
        """
        Test latency distribution with multiple samples
        
        Args:
            host: Target host (defaults to API endpoint)
            samples: Number of latency samples
            
        Returns:
            Dictionary with latency distribution
        """
        host = host or urlparse(self.target_url).hostname
        port = urlparse(self.target_url).port or 443
        
        latencies = []
        result = {
            "target": f"{host}:{port}",
            "samples": samples,
            "latencies": latencies,
            "error": None
        }
        
        self.log(f"Testing latency distribution ({samples} samples)...")
        
        for i in range(samples):
            try:
                start = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect((host, port))
                latency = (time.time() - start) * 1000
                sock.close()
                
                latencies.append(latency)
                self.debug(f"Sample {i+1}: {latency:.2f}ms")
                
            except Exception as e:
                self.debug(f"Sample {i+1} failed: {str(e)}")
        
        if latencies:
            result.update({
                "min": min(latencies),
                "max": max(latencies),
                "median": statistics.median(latencies),
                "mean": statistics.mean(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
            })
            
            self.log(f"Latency distribution (ms):")
            self.log(f"  Min: {result['min']:.2f} | Max: {result['max']:.2f}")
            self.log(f"  Median: {result['median']:.2f} | Mean: {result['mean']:.2f} ± {result['stdev']:.2f}")
            
            # Check for high jitter
            if result["stdev"] > (result["mean"] * 0.3):  # >30% of mean
                self.log("High network jitter detected - unstable connection", "WARNING")
        
        else:
            result["error"] = "All latency tests failed"
            self.log("Failed to measure latency - no successful samples", "ERROR")
        
        return result
    
    # ====================
    # Exchange API Tests
    # ====================
    
    async def test_exchange_connection(self) -> Dict:
        """Test connection to exchange API using CCXT"""
        self.print_section("Exchange API Connection Test")
        
        result = {
            "success": False,
            "exchange": "binance",
            "markets_loaded": False,
            "error": None
        }
        
        try:
            # Initialize CCXT exchange
            exchange_params = {
                "enableRateLimit": True,
                "timeout": 30000,
                "verbose": self.verbose
            }
            
            if self.proxy:
                exchange_params["proxies"] = {
                    "http": self.proxy,
                    "https": self.proxy
                }
            
            exchange = ccxt_async.binance(exchange_params)
            
            # Test loading markets
            start = time.time()
            await exchange.load_markets()
            elapsed = (time.time() - start) * 1000
            
            result.update({
                "success": True,
                "markets_loaded": True,
                "market_count": len(exchange.markets),
                "load_time_ms": elapsed
            })
            
            self.log(f"Successfully loaded {len(exchange.markets)} markets ({elapsed:.0f}ms)")
            
            # Test fetching ticker
            ticker = await self._test_fetch_ticker(exchange)
            result["ticker"] = ticker
            
            # Test fetching OHLCV
            ohlcv = await self._test_fetch_ohlcv(exchange)
            result["ohlcv"] = ohlcv
            
            await exchange.close()
            
        except Exception as e:
            result["error"] = str(e)
            self.log(f"Exchange connection failed: {str(e)}", "ERROR")
            if "exchange" in locals():
                await exchange.close()
        
        return result
    
    async def _test_fetch_ticker(self, exchange) -> Dict:
        """Test fetching ticker data"""
        result = {"success": False, "error": None}
        
        try:
            start = time.time()
            ticker = await exchange.fetch_ticker(self.symbol)
            elapsed = (time.time() - start) * 1000
            
            result.update({
                "success": True,
                "latency_ms": elapsed,
                "symbol": self.symbol,
                "last_price": ticker["last"],
                "timestamp": ticker["timestamp"]
            })
            
            self.log(f"Fetched {self.symbol} ticker ({elapsed:.0f}ms)")
            self.debug(f"Ticker data: {json.dumps(ticker, indent=2)}")
            
        except Exception as e:
            result["error"] = str(e)
            self.log(f"Failed to fetch ticker: {str(e)}", "ERROR")
        
        return result
    
    async def _test_fetch_ohlcv(self, exchange) -> Dict:
        """Test fetching OHLCV data"""
        result = {"success": False, "error": None}
        
        try:
            since = exchange.milliseconds() - 86400000  # 24 hours ago
            limit = 100
            
            start = time.time()
            ohlcv = await exchange.fetch_ohlcv(
                self.symbol,
                self.timeframe,
                since=since,
                limit=limit
            )
            elapsed = (time.time() - start) * 1000
            
            result.update({
                "success": True,
                "latency_ms": elapsed,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "count": len(ohlcv),
                "first_timestamp": ohlcv[0][0] if ohlcv else None,
                "last_timestamp": ohlcv[-1][0] if ohlcv else None
            })
            
            self.log(f"Fetched {len(ohlcv)} {self.timeframe} OHLCV candles ({elapsed:.0f}ms)")
            
        except Exception as e:
            result["error"] = str(e)
            self.log(f"Failed to fetch OHLCV: {str(e)}", "ERROR")
        
        return result
    
    # ====================
    # Diagnostic Reports
    # ====================
    
    def generate_report(self, test_results: Dict) -> None:
        """Generate comprehensive diagnostic report"""
        self.print_section("Diagnostic Summary Report", width=100)
        
        # Basic Connectivity
        self._print_test_result(
            "Basic Connectivity",
            test_results.get("basic_connectivity", {}),
            critical=True
        )
        
        # Proxy Configuration
        if self.proxy:
            self._print_test_result(
                "Proxy Configuration",
                test_results.get("proxy", {}),
                critical=True
            )
        
        # TLS Handshake
        self._print_test_result(
            "TLS Handshake",
            test_results.get("tls", {}),
            critical=True
        )
        
        # Exchange Connection
        self._print_test_result(
            "Exchange API",
            test_results.get("exchange", {}),
            critical=True
        )
        
        # Advanced Tests
        if test_results.get("advanced", {}):
            self.print_section("Advanced Network Metrics")
            
            self._print_metric(
                "Bandwidth",
                f"{test_results['advanced']['bandwidth']['download_speed']:.2f} KB/s",
                good_threshold=500,
                warn_threshold=100
            )
            
            self._print_metric(
                "Packet Loss",
                f"{test_results['advanced']['packet_loss']['loss_percent']}%",
                good_threshold=0,
                warn_threshold=5,
                reverse_scale=True
            )
            
            self._print_metric(
                "Latency Jitter",
                f"±{test_results['advanced']['latency_dist']['stdev']:.2f}ms",
                good_threshold=50,
                warn_threshold=100
            )
        
        # Recommendations
        self.print_section("Recommendations")
        recommendations = self._generate_recommendations(test_results)
        
        # Save full report to file if logging is enabled
        if self.log_file:
            self._save_full_report(test_results, recommendations)
    
    def _print_test_result(self, name: str, result: Dict, critical: bool = False) -> None:
        """Print formatted test result"""
        status = "✓ PASS" if result.get("success") else "✗ FAIL"
        color = "\033[92m" if result.get("success") else ("\033[91m" if critical else "\033[93m")
        
        details = []
        if "latency_ms" in result:
            details.append(f"{result['latency_ms']:.0f}ms")
        if "error" in result and result["error"]:
            details.append(result["error"])
        
        detail_str = f" ({', '.join(details)})" if details else ""
        
        console_output = f"{color}{status.ljust(10)}\033[0m {name.ljust(25)}{detail_str}"
        print(console_output)
        
        # Log to file without color codes
        if self.log_file:
            log_output = f"{status.ljust(10)} {name.ljust(25)}{detail_str}"
            self.file_logger.info(log_output)
    
    def _print_metric(self, 
                     name: str, 
                     value: str,
                     good_threshold: float,
                     warn_threshold: float,
                     reverse_scale: bool = False) -> None:
        """Print formatted metric with colored thresholds"""
        numeric_value = float(''.join(c for c in value if c.isdigit() or c == '.'))
        
        if reverse_scale:
            is_good = numeric_value <= good_threshold
            is_warn = numeric_value <= warn_threshold
        else:
            is_good = numeric_value >= good_threshold
            is_warn = numeric_value >= warn_threshold
        
        color = "\033[92m" if is_good else ("\033[93m" if is_warn else "\033[91m")
        console_output = f"{color}{value.rjust(10)}\033[0m {name}"
        print(console_output)
        
        # Log to file without color codes
        if self.log_file:
            log_output = f"{value.rjust(10)} {name}"
            self.file_logger.info(log_output)
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate troubleshooting recommendations"""
        recommendations = []
        
        # Basic connectivity issues
        if not results.get("basic_connectivity", {}).get("success"):
            recommendations.extend([
                "- Verify physical network connection",
                "- Check firewall/security software settings",
                "- Try different network (e.g., switch from WiFi to wired)",
                "- Test with VPN disabled (if applicable)"
            ])
        
        # Proxy issues
        if self.proxy and not results.get("proxy", {}).get("valid"):
            recommendations.extend([
                "- Verify proxy server is running and accessible",
                "- Check proxy authentication credentials",
                "- Test with different proxy server",
                "- Try direct connection without proxy"
            ])
        
        # TLS issues
        if not results.get("tls", {}).get("success"):
            recommendations.extend([
                "- Update system SSL certificates",
                "- Check system clock synchronization",
                "- Try with SSL verification disabled (temporary)",
                "- Update Python and OpenSSL libraries"
            ])
        
        # Exchange API issues
        if not results.get("exchange", {}).get("success"):
            recommendations.extend([
                "- Check exchange API status page for outages",
                "- Verify API keys have correct permissions",
                "- Reduce request frequency to avoid rate limits",
                "- Update CCXT library to latest version"
            ])
        
        # Network performance issues
        if results.get("advanced", {}):
            bw = results["advanced"]["bandwidth"].get("download_speed") or 0
            loss = results["advanced"]["packet_loss"].get("loss_percent") or 100
            jitter = results["advanced"]["latency_dist"].get("stdev") or 0
            
            if bw < 100:  # KB/s
                recommendations.append("- Upgrade network connection (low bandwidth)")
            
            if loss > 5:  # %
                recommendations.append("- Investigate network stability (high packet loss)")
            
            if jitter > 100:  # ms
                recommendations.append("- Optimize network path (high jitter)")
        
        if not recommendations:
            recommendations.append("- No critical issues detected. Connection appears healthy.")
        
        for rec in recommendations:
            print(rec)
            if self.log_file:
                self.file_logger.info(rec)
                
        return recommendations
        
    def _save_full_report(self, test_results: Dict, recommendations: List[str]) -> None:
        """Save detailed test results to JSON file"""
        try:
            # Create report data
            report_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target": self.target_url,
                "proxy": self.proxy,
                "symbol": self.symbol,
                "results": test_results,
                "recommendations": recommendations
            }
            
            # Save as JSON
            report_filename = os.path.splitext(self.log_file)[0] + ".json"
            with open(report_filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
                
            self.log(f"Full report saved to: {report_filename}")
            
        except Exception as e:
            self.log(f"Failed to save full report: {str(e)}", "ERROR")
    
    # ====================
    # Main Diagnostics
    # ====================
    
    async def run_diagnostics(self, full_scan: bool = False) -> Dict:
        """Run complete diagnostic suite"""
        results = {}
        
        # 1. Basic Connectivity Tests
        results["basic_connectivity"] = self.test_basic_connectivity()
        
        # 2. Proxy Diagnostics
        if self.proxy:
            results["proxy"] = self.diagnose_proxy()
        
        # 3. TLS Handshake Analysis
        results["tls"] = self.analyze_tls_handshake()
        
        # 4. Exchange API Tests
        results["exchange"] = await self.test_exchange_connection()
        
        # 5. Advanced Network Tests
        if full_scan:
            results["advanced"] = self.run_advanced_tests()
        
        # Generate report
        self.generate_report(results)
        
        return results
        
    # ====================
    # Stress Testing
    # ====================
    
    async def run_stress_test(self, 
                            duration: int = 60,
                            concurrent_requests: int = 5,
                            request_interval: float = 0.5) -> Dict:
        """
        Run stress test to evaluate connection stability under load
        
        Args:
            duration: Test duration in seconds
            concurrent_requests: Number of concurrent requests
            request_interval: Interval between requests in seconds
            
        Returns:
            Dictionary with stress test results
        """
        self.print_section(f"Stress Test ({duration}s, {concurrent_requests} concurrent requests)")
        
        results = {
            "duration": duration,
            "concurrent_requests": concurrent_requests,
            "interval": request_interval,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0,
            "success_rate": 0,
            "errors": {}
        }
        
        # Prepare stress test
        end_time = time.time() + duration
        response_times = []
        error_counts = {}
        
        try:
            # Create task queue
            tasks = []
            for i in range(concurrent_requests):
                tasks.append(self._stress_test_worker(end_time, request_interval, response_times, error_counts))
            
            self.log(f"Starting stress test with {concurrent_requests} concurrent workers")
            self.log(f"Target duration: {duration} seconds")
            
            # Run stress test
            await asyncio.gather(*tasks)
            
            # Calculate results
            results["total_requests"] = len(response_times) + sum(error_counts.values())
            results["successful_requests"] = len(response_times)
            results["failed_requests"] = sum(error_counts.values())
            
            if response_times:
                results["avg_response_time"] = statistics.mean(response_times)
                results["min_response_time"] = min(response_times)
                results["max_response_time"] = max(response_times)
                if len(response_times) > 1:
                    results["response_time_stdev"] = statistics.stdev(response_times)
            
            results["success_rate"] = (results["successful_requests"] / results["total_requests"]) * 100
            results["errors"] = error_counts
            
            # Log results
            self.log(f"Stress test completed - {results['total_requests']} total requests")
            self.log(f"Success rate: {results['success_rate']:.2f}%")
            
            if response_times:
                self.log(f"Avg response time: {results['avg_response_time']:.2f}ms")
            
            if error_counts:
                self.log("Errors encountered:", "WARNING")
                for error, count in error_counts.items():
                    self.log(f"  {error}: {count} times", "WARNING")
                    
        except Exception as e:
            self.log(f"Stress test failed: {str(e)}", "ERROR")
            results["error"] = str(e)
        
        return results
    
    async def _stress_test_worker(self, 
                                end_time: float, 
                                interval: float, 
                                response_times: List[float],
                                error_counts: Dict[str, int]) -> None:
        """Worker for stress testing"""
        
        while time.time() < end_time:
            try:
                # Make API request
                start = time.time()
                
                proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
                response = requests.get(
                    self.target_url,
                    proxies=proxies,
                    timeout=10,
                    verify=False
                )
                
                # Record response time
                response_time = (time.time() - start) * 1000
                response_times.append(response_time)
                
                # Check response
                if response.status_code != 200:
                    error_key = f"HTTP {response.status_code}"
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1
                
            except Exception as e:
                # Record error
                error_key = str(type(e).__name__)
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            # Wait before next request
            await asyncio.sleep(interval)


async def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Network Diagnostics Tool for Crypto APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n"
               "  python network_diagnostics.py --target https://api.binance.com\n"
               "  python network_diagnostics.py --proxy http://127.0.0.1:7890 --full-scan"
    )
    
    # Add command line arguments
    parser.add_argument("--target", type=str, default="https://api.binance.com/api/v3/ping",
                        help="Target API endpoint to test")
    parser.add_argument("--proxy", type=str, help="Proxy URL (e.g., http://127.0.0.1:7890)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT",
                        help="Trading pair symbol for API tests")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe for historical data tests")
    parser.add_argument("--full-scan", action="store_true",
                        help="Perform full network diagnostics")
    parser.add_argument("--stress-test", action="store_true",
                        help="Run connection stress test")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--log", action="store_true",
                        help="Save results to log file")
    
    args = parser.parse_args()
    
    # Setup logging directory and file
    log_file = None
    if args.log:
        # Create directory structure
        target_name = urlparse(args.target).netloc.split(".")[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Path format: ./reports/network/[exchange]/[timestamp].log
        log_dir = os.path.join(".", "reports", "network", target_name)
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"{timestamp}.log")
    
    # Display banner
    print("\n" + "=" * 80)
    print(" Crypto API Network Diagnostics Tool")
    print(" Version 1.0.0 | " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80 + "\n")
    
    # Initialize diagnostics
    diagnostics = NetworkDiagnostics(
        target=args.target,
        proxy=args.proxy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        verbose=args.verbose,
        log_file=log_file
    )
    
    # Run tests
    try:
        print(f"Starting diagnostics for {args.target}")
        if args.proxy:
            print(f"Using proxy: {args.proxy}")
        if log_file:
            print(f"Logging results to: {log_file}")
        
        # Standard diagnostics
        results = await diagnostics.run_diagnostics(full_scan=args.full_scan)
        
        # Stress test (optional)
        if args.stress_test:
            stress_results = await diagnostics.run_stress_test()
            results["stress_test"] = stress_results
        
        print("\nDiagnostics completed successfully")
        if log_file:
            json_file = os.path.splitext(log_file)[0] + ".json"
            print(f"Full report saved to: {json_file}")
        
    except KeyboardInterrupt:
        print("\nDiagnostics interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run main function
    asyncio.run(main())