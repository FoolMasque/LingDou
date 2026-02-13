import requests
import json
import time
import os
from typing import List, Dict, Optional, Any

class WeChatShopConnector:
    """
    WeChat Shop API Connector
    Handles Access Token management and API calls
    """
    def __init__(self, token_url: str):
        self.token_url = token_url
        self.access_token = None
        self.token_expiry = 0
        self.BASE_URL = "https://api.weixin.qq.com"

    def _get_access_token(self) -> str:
        """
        Get or refresh access token from internal service
        """
        # Simple cache: refresh if expired or not set
        if self.access_token and time.time() < self.token_expiry:
            return self.access_token

        print(f"Fetching new access token from: {self.token_url}...")
        try:
            resp = requests.get(self.token_url, timeout=10)
            
            # The user's example response wasn't shown, but typically it returns a JSON with the token
            # Assuming standard JSON response: {"access_token": "...", "expires_in": 7200}
            # OR simple string response? Let's handle both or try to parse JSON
            
            try:
                data = resp.json()
                # Adjust key based on actual response format, assuming 'data' or direct 'access_token'
                # If the response is standard WeChat format wrapped:
                token = data.get("access_token") or data.get("token") or data.get("data")
                expires_in = data.get("expires_in", 3600)
            except:
                # Fallback: treat content as token string
                token = resp.text.strip()
                expires_in = 3600

            if not token:
                raise Exception(f"Empty token received from {self.token_url}")

            self.access_token = token
            self.token_expiry = time.time() + expires_in - 200
            print("Access token refreshed successfully.")
            return self.access_token

        except Exception as e:
            print(f"Error fetching token: {e}")
            raise

    def get_product_ids(self) -> List[str]:
        """
        Fetch ALL product IDs from the shop
        Handles pagination automatically
        """
        token = self._get_access_token()
        url = f"{self.BASE_URL}/channels/ec/product/list/get"
        params = {"access_token": token}
        
        all_ids = []
        next_key = None
        
        print("Starting to fetch product list...")
        
        while True:
            payload = {
                "page_size": 100,  # Max allowed usually
                "next_key": next_key
            } if next_key else {"page_size": 100}
            
            try:
                resp = requests.post(url, params=params, json=payload, timeout=20)
                data = resp.json()
                
                if data.get("errcode") != 0:
                    print(f"Error fetching list: {data}")
                    break
                
                product_ids = data.get("product_ids", [])
                all_ids.extend([str(pid) for pid in product_ids]) # Ensure strings
                print(f"Fetched {len(product_ids)} items. Total so far: {len(all_ids)}")
                
                next_key = data.get("next_key")
                
                # Check for end of list conditions
                if not next_key or len(product_ids) == 0:
                    break
                    
                # Rate limit protection
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Exception during list fetch: {e}")
                break
                
        return all_ids

    def get_product_detail(self, product_id: str) -> Optional[Dict]:
        """
        Fetch details for a single product
        """
        token = self._get_access_token()
        url = f"{self.BASE_URL}/channels/ec/product/get"
        params = {"access_token": token}
        
        payload = {
            "product_id": str(product_id),
            "data_type": 1
        }
        
        try:
            resp = requests.post(url, params=params, json=payload, timeout=10)
            data = resp.json()
            
            if data.get("errcode") != 0:
                print(f"Error fetching detail for {product_id}: {data.get('errmsg')}")
                return None
                
            return data.get("product")
            
        except Exception as e:
            print(f"Exception fetching detail for {product_id}: {e}")
            return None

if __name__ == "__main__":
    # Test stub
    print("This is a module, import it to use.")
