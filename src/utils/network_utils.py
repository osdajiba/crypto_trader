# src/utils/network.py
import os
import socket
import requests
from typing import Dict, Optional, Union
import subprocess
import re
import platform
from urllib.parse import urlparse
from src.common.log_manager import LogManager


logger = LogManager.get_logger("data.manager")


class ProxyDetector:
    """
    Utility class for detecting and validating proxy configurations
    to use with external APIs like Binance.
    """
    
    def __init__(self, test_url: str = 'https://api.binance.com/api/v3/ping', timeout: int = 5):
        """
        Initialize the proxy detector
        
        Args:
            test_url: URL to use for testing proxy connections
            timeout: Connection timeout in seconds
        """
        self.test_url = test_url
        self.timeout = timeout
        self.common_proxy_ports = [7890, 8080, 1080, 8118, 3128, 10809, 7070, 9090]
    
    def detect_proxy(self, check_env: bool = True, check_system: bool = True, 
                   check_common_ports: bool = True) -> Optional[Dict[str, str]]:
        """
        Detect available proxy configuration using multiple methods
        
        Args:
            check_env: Whether to check environment variables
            check_system: Whether to check system proxy settings
            check_common_ports: Whether to check common proxy ports
            
        Returns:
            Dict containing http/https proxy URLs or None if no working proxy found
        """
        proxy_sources = []
        
        # 1. Check environment variables
        if check_env:
            env_proxies = self._check_environment_variables()
            if env_proxies:
                proxy_sources.append(("Environment Variables", env_proxies))
        
        # 2. Check system proxy settings
        if check_system:
            system_proxies = self._check_system_proxy()
            if system_proxies:
                proxy_sources.append(("System Settings", system_proxies))
        
        # 3. Check requests library's auto-detection
        requests_proxies = self._check_requests_proxy()
        if requests_proxies:
            proxy_sources.append(("Requests Auto-detection", requests_proxies))
        
        # 4. Check common proxy ports
        if check_common_ports:
            for port in self.common_proxy_ports:
                common_proxy = self._check_common_proxy_port(port)
                if common_proxy:
                    proxy_sources.append((f"Common Port {port}", common_proxy))
                    break
        
        # Test each detected proxy and return the first working one
        for source, proxy in proxy_sources:
            if self.test_proxy(proxy):
                logger.info(f"Found working proxy from {source}: {proxy}")
                return proxy
                
        logger.info("No working proxy found, using direct connection")
        return None
    
    def _check_environment_variables(self) -> Optional[Dict[str, str]]:
        """Check for proxy configuration in environment variables"""
        env_vars = {
            'http': ['HTTP_PROXY', 'http_proxy'],
            'https': ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy']
        }
        
        proxies = {}
        
        for protocol, var_names in env_vars.items():
            for var in var_names:
                if var in os.environ and os.environ[var]:
                    proxies[protocol] = os.environ[var]
                    break
        
        if proxies:
            logger.debug(f"Found proxy settings in environment variables: {proxies}")
            return proxies
        
        return None
    
    def _check_system_proxy(self) -> Optional[Dict[str, str]]:
        """Check for system proxy configuration using platform-specific methods"""
        proxies = None
        system = platform.system()
        
        try:
            if system == 'Windows':
                proxies = self._get_windows_proxy()
            elif system == 'Darwin':  # macOS
                proxies = self._get_macos_proxy()
            elif system == 'Linux':
                proxies = self._get_linux_proxy()
            
            if proxies:
                logger.debug(f"Found system proxy settings: {proxies}")
                return proxies
        except Exception as e:
            logger.debug(f"Error checking system proxy: {e}")
        
        return None
    
    def _get_windows_proxy(self) -> Optional[Dict[str, str]]:
        """Get proxy settings from Windows registry"""
        try:
            import winreg
            
            proxy_key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
            )
            
            # Check if proxy is enabled
            proxy_enable, _ = winreg.QueryValueEx(proxy_key, "ProxyEnable")
            
            if proxy_enable:
                proxy_server, _ = winreg.QueryValueEx(proxy_key, "ProxyServer")
                
                # Handle different proxy formats
                if "=" in proxy_server:  # Protocol-specific format
                    protocols = proxy_server.split(";")
                    proxies = {}
                    
                    for protocol in protocols:
                        if "=" in protocol:
                            proto_name, address = protocol.split("=", 1)
                            if proto_name.lower() in ('http', 'https'):
                                proxies[proto_name.lower()] = address
                    
                    return proxies if proxies else None
                else:  # Same proxy for all protocols
                    return {'http': proxy_server, 'https': proxy_server}
            
        except Exception as e:
            logger.debug(f"Error reading Windows registry: {e}")
        
        return None
    
    def _get_macos_proxy(self) -> Optional[Dict[str, str]]:
        """Get proxy settings from macOS system preferences"""
        try:
            # Check HTTP proxy
            result = subprocess.run(
                ['networksetup', '-getwebproxy', 'Wi-Fi'],
                capture_output=True, text=True
            )
            
            http_enabled = "Enabled: Yes" in result.stdout
            http_proxy = None
            
            if http_enabled:
                server_match = re.search(r'Server: (.*)', result.stdout)
                port_match = re.search(r'Port: (\d+)', result.stdout)
                
                if server_match and port_match:
                    server = server_match.group(1)
                    port = port_match.group(1)
                    http_proxy = f"http://{server}:{port}"
            
            # Check HTTPS proxy
            result = subprocess.run(
                ['networksetup', '-getsecurewebproxy', 'Wi-Fi'],
                capture_output=True, text=True
            )
            
            https_enabled = "Enabled: Yes" in result.stdout
            https_proxy = None
            
            if https_enabled:
                server_match = re.search(r'Server: (.*)', result.stdout)
                port_match = re.search(r'Port: (\d+)', result.stdout)
                
                if server_match and port_match:
                    server = server_match.group(1)
                    port = port_match.group(1)
                    https_proxy = f"http://{server}:{port}"
            
            if http_proxy or https_proxy:
                return {
                    'http': http_proxy,
                    'https': https_proxy or http_proxy
                }
            
        except Exception as e:
            logger.debug(f"Error checking macOS proxy: {e}")
        
        return None
    
    def _get_linux_proxy(self) -> Optional[Dict[str, str]]:
        """Get proxy settings from Linux environments"""
        # Check common environment variables
        proxies = self._check_environment_variables()
        if proxies:
            return proxies
            
        # Check GNOME settings
        try:
            result = subprocess.run(
                ['gsettings', 'get', 'org.gnome.system.proxy', 'mode'],
                capture_output=True, text=True
            )
            
            if 'manual' in result.stdout:
                http_host = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.http', 'host'],
                    capture_output=True, text=True
                ).stdout.strip().strip("'")
                
                http_port = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.http', 'port'],
                    capture_output=True, text=True
                ).stdout.strip()
                
                https_host = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.https', 'host'],
                    capture_output=True, text=True
                ).stdout.strip().strip("'")
                
                https_port = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.https', 'port'],
                    capture_output=True, text=True
                ).stdout.strip()
                
                proxies = {}
                
                if http_host and http_port:
                    proxies['http'] = f"http://{http_host}:{http_port}"
                
                if https_host and https_port:
                    proxies['https'] = f"http://{https_host}:{https_port}"
                elif 'http' in proxies:
                    proxies['https'] = proxies['http']
                
                if proxies:
                    return proxies
        
        except Exception as e:
            logger.debug(f"Error checking Linux proxy settings: {e}")
        
        return None
    
    def _check_requests_proxy(self) -> Optional[Dict[str, str]]:
        """Check for proxy settings using requests library's auto-detection"""
        try:
            session = requests.Session()
            proxies = session.proxies
            
            # Filter out empty proxies
            proxies = {k: v for k, v in proxies.items() if v}
            
            if proxies:
                logger.debug(f"Found proxy settings via requests auto-detection: {proxies}")
                return proxies
        except Exception as e:
            logger.debug(f"Error checking requests proxies: {e}")
        
        return None
    
    def _check_common_proxy_port(self, port: int) -> Optional[Dict[str, str]]:
        """Check if a common proxy port is available on localhost"""
        proxy_url = f"http://127.0.0.1:{port}"
        
        try:
            # Try to connect to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result == 0:  # Port is open
                logger.debug(f"Found open proxy port at: {proxy_url}")
                return {'http': proxy_url, 'https': proxy_url}
        except Exception as e:
            logger.debug(f"Error checking proxy port {port}: {e}")
        
        return None
    
    def test_proxy(self, proxy_settings: Dict[str, str]) -> bool:
        """
        Test if a proxy configuration works with the target API
        
        Args:
            proxy_settings: Dict with 'http' and/or 'https' proxy URLs
            
        Returns:
            bool: True if proxy works, False otherwise
        """
        if not proxy_settings:
            return False
            
        try:
            logger.debug(f"Testing proxy configuration: {proxy_settings}")
            response = requests.get(
                self.test_url,
                proxies=proxy_settings,
                timeout=self.timeout
            )
            
            success = response.status_code == 200
            logger.debug(f"Proxy test {'successful' if success else 'failed'} with status {response.status_code}")
            return success
        except Exception as e:
            logger.debug(f"Proxy test failed with error: {e}")
            return False
    
    def get_proxy_info(self, proxy_url: str) -> Dict[str, Union[str, int]]:
        """
        Parse proxy URL and extract components
        
        Args:
            proxy_url: Proxy URL string
            
        Returns:
            Dict containing protocol, host, port, and auth info
        """
        if not proxy_url:
            return {}
            
        try:
            parsed = urlparse(proxy_url)
            
            # Handle auth info if present
            auth = None
            if parsed.username and parsed.password:
                auth = f"{parsed.username}:{parsed.password}"
            
            return {
                'protocol': parsed.scheme,
                'host': parsed.hostname,
                'port': parsed.port,
                'auth': auth
            }
        except Exception as e:
            logger.error(f"Error parsing proxy URL: {e}")
            return {}

def detect_proxy(test_url: str = 'https://api.binance.com/api/v3/ping') -> Optional[Dict[str, str]]:
    """
    Convenience function to detect available proxy settings for the target URL
    
    Args:
        test_url: URL to test proxy against
        
    Returns:
        Dict with proxy configuration or None if no working proxy found
    """
    detector = ProxyDetector(test_url=test_url)
    return detector.detect_proxy()

def test_connection(url: str = 'https://api.binance.com/api/v3/ping', 
                   proxy: Optional[Dict[str, str]] = None, 
                   timeout: int = 5) -> bool:
    """
    Test connection to a URL with optional proxy
    
    Args:
        url: URL to test
        proxy: Optional proxy configuration
        timeout: Connection timeout in seconds
        
    Returns:
        bool: True if connection succeeded, False otherwise
    """
    try:
        response = requests.get(url, proxies=proxy, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.debug(f"Connection test failed: {e}")
        return False