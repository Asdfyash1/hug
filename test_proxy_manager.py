"""
Quick test for ProxyManager and /proxy endpoint
"""
import sys
import os
sys.path.append(os.getcwd())

from api.proxy_manager import ProxyManager

def test_proxy_manager():
    print("=" * 60)
    print("Testing ProxyManager")
    print("=" * 60)
    
    # Test 1: Single proxy
    print("\n[Test 1] Single proxy format (ip:port:user:pass)")
    pm = ProxyManager(proxy_url="31.59.20.176:6754:nntlrciu:sx2noxvkj6y7")
    
    proxy = pm.get_next_proxy()
    print(f"Got proxy: {proxy}")
    assert proxy == "http://nntlrciu:sx2noxvkj6y7@31.59.20.176:6754"
    print("[OK] Proxy format conversion works!")
    
    # Test 2: Bandwidth tracking
    print("\n[Test 2] Bandwidth tracking")
    pm.track_bandwidth(proxy, 50000)  # 50KB
    stats = pm.get_stats()
    print(f"Total bandwidth: {stats['total_bandwidth_bytes']} bytes")
    print(f"Total bandwidth: {stats['total_bandwidth_mb']} MB")
    assert stats['total_bandwidth_bytes'] == 50000
    print("[OK] Bandwidth tracking works!")
    
    # Test 3: Proxy rotation
    print("\n[Test 3] Proxy rotation")
    pm.add_proxy("23.95.150.145:6114:nntlrciu:sx2noxvkj6y7")
    pm.add_proxy("198.23.239.134:6540:nntlrciu:sx2noxvkj6y7")
    
    proxies_used = []
    for i in range(6):
        p = pm.get_next_proxy()
        proxies_used.append(p.split('@')[1] if '@' in p else p)
    
    print(f"Rotation order: {proxies_used}")
    # Should rotate: proxy1, proxy2, proxy3, proxy1, proxy2, proxy3
    assert proxies_used[0] == proxies_used[3]
    assert proxies_used[1] == proxies_used[4]
    print("[OK] Proxy rotation works!")
    
    # Test 4: Stats
    print("\n[Test 4] Stats display")
    stats = pm.get_stats()
    print(f"Total proxies: {stats['total_proxies']}")
    print(f"Current index: {stats['current_proxy_index']}")
    print(f"Bandwidth remaining: {stats['bandwidth_remaining_mb']} MB")
    assert stats['total_proxies'] == 3
    print("[OK] Stats work!")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All ProxyManager tests passed!")
    print("=" * 60)
    
    # Cleanup
    if os.path.exists("proxy_stats.json"):
        os.remove("proxy_stats.json")

if __name__ == '__main__':
    test_proxy_manager()
