"""
Proxy Manager for Viral Clip Extractor
Handles proxy rotation, WebShare API integration, and bandwidth tracking
"""
import os
import json
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

class ProxyManager:
    def __init__(self, proxy_url: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize ProxyManager
        
        Args:
            proxy_url: Single proxy in format 'http://user:pass@host:port' or 'host:port:user:pass'
            api_url: WebShare API URL for dynamic proxy list
        """
        self.proxies: List[str] = []
        self.current_index = 0
        self.bandwidth_stats: Dict[str, int] = {}
        self.total_bandwidth = 0
        self.stats_file = "proxy_stats.json"
        self.lock = threading.Lock()
        
        # Load existing stats
        self._load_stats()
        
        # Initialize proxies
        if api_url:
            logger.info("Initializing with WebShare API")
            self._fetch_from_api(api_url)
        elif proxy_url:
            logger.info("Initializing with single proxy")
            self._add_single_proxy(proxy_url)
        else:
            logger.warning("No proxy configuration provided")
    
    def _parse_proxy_format(self, proxy_str: str) -> str:
        """
        Parse proxy from 'host:port:user:pass' to 'http://user:pass@host:port'
        Also handles already formatted proxies
        """
        proxy_str = proxy_str.strip()
        
        # Already in correct format
        if proxy_str.startswith('http://') or proxy_str.startswith('https://'):
            return proxy_str
        
        # Parse host:port:user:pass format
        parts = proxy_str.split(':')
        if len(parts) == 4:
            host, port, user, password = parts
            return f"http://{user}:{password}@{host}:{port}"
        elif len(parts) == 2:
            # Just host:port (no auth)
            host, port = parts
            return f"http://{host}:{port}"
        else:
            logger.error(f"Invalid proxy format: {proxy_str}")
            return proxy_str
    
    def _add_single_proxy(self, proxy_url: str):
        """Add a single proxy"""
        formatted = self._parse_proxy_format(proxy_url)
        self.proxies.append(formatted)
        if formatted not in self.bandwidth_stats:
            self.bandwidth_stats[formatted] = 0
    
    def _fetch_from_api(self, api_url: str):
        """Fetch proxy list from WebShare API"""
        try:
            logger.info(f"Fetching proxies from API: {api_url[:50]}...")
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            # Parse response - WebShare returns one proxy per line
            proxy_list = response.text.strip().split('\n')
            
            for proxy_line in proxy_list:
                if proxy_line.strip():
                    formatted = self._parse_proxy_format(proxy_line)
                    self.proxies.append(formatted)
                    if formatted not in self.bandwidth_stats:
                        self.bandwidth_stats[formatted] = 0
            
            logger.info(f"Loaded {len(self.proxies)} proxies from API")
        except Exception as e:
            logger.error(f"Failed to fetch proxies from API: {e}")
    
    def get_next_proxy(self) -> Optional[str]:
        """Get next proxy using round-robin rotation"""
        with self.lock:
            if not self.proxies:
                return None
            
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            return proxy
    
    def track_bandwidth(self, proxy: str, bytes_used: int):
        """Track bandwidth usage for a proxy"""
        with self.lock:
            if proxy in self.bandwidth_stats:
                self.bandwidth_stats[proxy] += bytes_used
            else:
                self.bandwidth_stats[proxy] = bytes_used
            
            self.total_bandwidth += bytes_used
            self._save_stats()
    
    def get_stats(self) -> Dict:
        """Get bandwidth statistics"""
        with self.lock:
            return {
                "total_proxies": len(self.proxies),
                "total_bandwidth_bytes": self.total_bandwidth,
                "total_bandwidth_mb": round(self.total_bandwidth / (1024 * 1024), 2),
                "total_bandwidth_gb": round(self.total_bandwidth / (1024 * 1024 * 1024), 3),
                "bandwidth_remaining_mb": max(0, 1024 - round(self.total_bandwidth / (1024 * 1024), 2)),
                "per_proxy_stats": [
                    {
                        "proxy": self._mask_proxy(proxy),
                        "bandwidth_bytes": bytes_used,
                        "bandwidth_mb": round(bytes_used / (1024 * 1024), 2)
                    }
                    for proxy, bytes_used in self.bandwidth_stats.items()
                ],
                "current_proxy_index": self.current_index,
                "last_updated": datetime.now().isoformat()
            }
    
    def _mask_proxy(self, proxy: str) -> str:
        """Mask proxy credentials for display"""
        if '@' in proxy:
            parts = proxy.split('@')
            return f"***@{parts[1]}"
        return proxy
    
    def _load_stats(self):
        """Load stats from file"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    self.bandwidth_stats = data.get('bandwidth_stats', {})
                    self.total_bandwidth = data.get('total_bandwidth', 0)
                    logger.info(f"Loaded stats: {self.total_bandwidth} bytes total")
        except Exception as e:
            logger.error(f"Failed to load stats: {e}")
    
    def _save_stats(self):
        """Save stats to file"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump({
                    'bandwidth_stats': self.bandwidth_stats,
                    'total_bandwidth': self.total_bandwidth,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
    
    def refresh_proxies(self, api_url: str):
        """Refresh proxy list from API"""
        with self.lock:
            old_count = len(self.proxies)
            self.proxies.clear()
            self._fetch_from_api(api_url)
            logger.info(f"Refreshed proxies: {old_count} -> {len(self.proxies)}")
    
    def add_proxy(self, proxy_str: str):
        """Add a new proxy"""
        with self.lock:
            formatted = self._parse_proxy_format(proxy_str)
            if formatted not in self.proxies:
                self.proxies.append(formatted)
                if formatted not in self.bandwidth_stats:
                    self.bandwidth_stats[formatted] = 0
                logger.info(f"Added proxy: {self._mask_proxy(formatted)}")
