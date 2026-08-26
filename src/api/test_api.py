import sys
import os
import json

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.app import app

def test_all_endpoints():
    client = app.test_client()
    
    print("=" * 60)
    print("TESTING /api/health")
    print("=" * 60)
    res_health = client.get('/api/health')
    print(f"Status Code: {res_health.status_code}")
    print(f"Response: {json.dumps(res_health.get_json(), indent=2)}")
    assert res_health.status_code == 200
    
    print("\n" + "=" * 60)
    print("TESTING /api/aqi/Lahore")
    print("=" * 60)
    res_aqi = client.get('/api/aqi/Lahore')
    print(f"Status Code: {res_aqi.status_code}")
    print(f"Response: {json.dumps(res_aqi.get_json(), indent=2)}")
    assert res_aqi.status_code == 200
    
    print("\n" + "=" * 60)
    print("TESTING /api/prediction/Lahore")
    print("=" * 60)
    res_pred = client.get('/api/prediction/Lahore')
    print(f"Status Code: {res_pred.status_code}")
    print(f"Response: {json.dumps(res_pred.get_json(), indent=2)}")
    assert res_pred.status_code == 200

    print("\n" + "=" * 60)
    print("TESTING /api/forecast/Lahore")
    print("=" * 60)
    res_fc = client.get('/api/forecast/Lahore')
    print(f"Status Code: {res_fc.status_code}")
    fc_json = res_fc.get_json()
    print(f"Status: {fc_json.get('status')}")
    print(f"City: {fc_json.get('data', {}).get('city')}")
    print(f"Forecast Hours: {fc_json.get('data', {}).get('forecast_hours')}")
    print(f"First 2 forecast steps: {json.dumps(fc_json.get('data', {}).get('forecast')[:2], indent=2)}")
    assert res_fc.status_code == 200
    assert fc_json.get('data', {}).get('forecast_hours') == 72
    
    print("\n" + "=" * 60)
    print("ALL 4 API ENDPOINTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_all_endpoints()
