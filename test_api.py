"""
Test suite for Viral Clip Extractor API
Run with: python -m pytest test_api.py
"""

import pytest
import json
from api.index import app, extract_video_id, time_to_seconds, seconds_to_time

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test health check"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'

def test_index_endpoint(client):
    """Test API info endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'name' in data
    assert 'endpoints' in data

def test_extract_video_id():
    """Test video ID extraction from various URLs"""
    # Standard URL
    assert extract_video_id('https://youtube.com/watch?v=dQw4w9WgXcQ') == 'dQw4w9WgXcQ'
    # Short URL
    assert extract_video_id('https://youtu.be/dQw4w9WgXcQ') == 'dQw4w9WgXcQ'
    # Embed URL
    assert extract_video_id('https://youtube.com/embed/dQw4w9WgXcQ') == 'dQw4w9WgXcQ'
    # Shorts URL
    assert extract_video_id('https://youtube.com/shorts/dQw4w9WgXcQ') == 'dQw4w9WgXcQ'
    # Just ID
    assert extract_video_id('dQw4w9WgXcQ') == 'dQw4w9WgXcQ'
    # Invalid
    assert extract_video_id('not-a-valid-url') is None

def test_time_conversions():
    """Test time conversion functions"""
    # Seconds to time
    assert seconds_to_time(120) == '2:00'
    assert seconds_to_time(65) == '1:05'
    assert seconds_to_time(0) == '0:00'
    
    # Time to seconds
    assert time_to_seconds('2:00') == 120
    assert time_to_seconds('1:30') == 90
    assert time_to_seconds('1:30:45') == 5445
    assert time_to_seconds(120) == 120

def test_extract_endpoint_missing_url(client):
    """Test extract endpoint with missing URL"""
    response = client.post('/extract', 
                          data=json.dumps({}),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_extract_endpoint_invalid_url(client):
    """Test extract endpoint with invalid URL"""
    response = client.post('/extract',
                          data=json.dumps({'url': 'invalid-url'}),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_analyze_endpoint_missing_url(client):
    """Test analyze endpoint with missing URL"""
    response = client.post('/analyze',
                          data=json.dumps({}),
                          content_type='application/json')
    assert response.status_code == 400

def test_score_endpoint_missing_params(client):
    """Test score endpoint with missing parameters"""
    response = client.post('/score',
                          data=json.dumps({'url': 'https://youtube.com/watch?v=test'}),
                          content_type='application/json')
    assert response.status_code == 400

def test_transcript_endpoint(client):
    """Test transcript endpoint structure"""
    response = client.post('/transcript',
                          data=json.dumps({
                              'url': 'https://youtube.com/watch?v=test',
                              'start': 0,
                              'end': 60
                          }),
                          content_type='application/json')
    # Will fail due to invalid video, but tests structure
    assert response.status_code in [400, 500]

def test_cors_headers(client):
    """Test CORS is enabled"""
    response = client.get('/')
    assert 'Access-Control-Allow-Origin' in response.headers

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
