import json
import os
import time
import requests
from typing import List, Dict, Set
from datetime import datetime

# Import local connector
try:
    from wechat_connector import WeChatShopConnector
except ImportError:
    # Handle relative import for script execution
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from wechat_connector import WeChatShopConnector

# Configuration
file_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(file_dir, "config.json")
STATE_FILE = os.path.join(file_dir, "synced_ids.json")
LINGDOU_API_URL = "http://localhost:8008/api/process_data"  # Default local API

class WeChatSyncService:
    def __init__(self):
        self.config = self._load_config()
        self.state = self._load_state()
        
        # Initialize connector
        self.connector = WeChatShopConnector(
            token_url=self.config.get("token_url")
        )
        self.business_id = self.config.get("business_id", "wechat_shop")

    def _load_config(self) -> Dict:
        if not os.path.exists(CONFIG_FILE):
             # Fallback or error
             print("Config file not found!")
             return {}
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_state(self) -> Set[str]:
        if not os.path.exists(STATE_FILE):
            return set()
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get("synced_ids", []))

    def _save_state(self):
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "last_sync": datetime.now().isoformat(),
                "synced_ids": list(self.state)
            }, f, indent=2)

    def _transform_product(self, raw_product: Dict) -> Dict:
        """
        Transform WeChat product to LingDou standard structure
        """
        # Extract main images
        head_imgs = raw_product.get("head_imgs", [])
        desc_imgs = raw_product.get("desc_info", {}).get("imgs", [])
        all_imgs = head_imgs + desc_imgs
        
        # Extract attributes
        attrs = {item["attr_key"]: item["attr_value"] for item in raw_product.get("attrs", [])}
        
        # Extract SKU info (get price range or first SKU price)
        skus = raw_product.get("skus", [])
        price = "未知"
        if skus:
            # Price is in cents
            prices = [sku.get("sale_price", 0) for sku in skus]
            if prices:
                min_p = min(prices) / 100
                max_p = max(prices) / 100
                price = f"{min_p}元" if min_p == max_p else f"{min_p}-{max_p}元"

        # Construct description
        desc_text = f"商品名: {raw_product.get('title')}\n"
        if raw_product.get('sub_title'):
             desc_text += f"副标题: {raw_product.get('sub_title')}\n"
        desc_text += f"价格: {price}\n"
        desc_text += "参数:\n" + "\n".join([f"- {k}: {v}" for k, v in attrs.items()])
        
        # Determine category/type
        cats = raw_product.get("cats_v2", [])
        cat_id = cats[-1].get("cat_id") if cats else "default"

        item = {
            "product_name": raw_product.get("title"),
            "product_id": raw_product.get("product_id"),
            "category": str(cat_id),
            "price": price,
            "description": desc_text,  # Full text description for embedding
            "specifications": attrs,   # Structured specs
            "detail_images": all_imgs,
            # Standard LingDou fields fallback
            "商品名": raw_product.get("title"),
            "价格": price,
            "参数": attrs
        }
        
        # Map specific image fields for LingDou
        if head_imgs:
            item["cover_pic"] = head_imgs[0]
        
        return item

    def run_sync(self):
        print(f"Starting sync for business: {self.business_id}")
        
        # 1. Get all remote IDs
        remote_ids = self.connector.get_product_ids()
        remote_ids_set = set(str(pid) for pid in remote_ids)
        print(f"Remote products found: {len(remote_ids_set)}")
        
        # 2. Identify new products
        new_ids = remote_ids_set - self.state
        
        if not new_ids:
            print("No new products to sync.")
            return

        print(f"Found {len(new_ids)} new products. Fetching details...")
        
        # 3. Fetch details and transform
        transformed_items = []
        synced_batch = set()
        
        for pid in new_ids:
            detail = self.connector.get_product_detail(pid)
            if detail:
                item = self._transform_product(detail)
                transformed_items.append(item)
                synced_batch.add(pid)
            else:
                print(f"Skipping {pid} due to fetch failure.")
            
            # Batch process? No, let's collect all for now or batch by 10
            # For simplicity, collect all then push, unless list is huge.
            # Let's verify batch size - user said "incremental", usually small.

        if not transformed_items:
            print("No valid details fetched.")
            return

        # 4. Push to LingDou API
        # Save to temp file first
        temp_file = f"import_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_items, f, indent=2, ensure_ascii=False)
            
        print(f"Prepared {len(transformed_items)} items in {temp_file}. Pushing to LingDou...")
        
        try:
            # LingDou API expects JSON with local file path: {"business_id": "...", "json_file": "..."}
            # Since we are running locally, we pass the absolute path.
            abs_file_path = os.path.abspath(temp_file)
            
            payload = {
                "business_id": self.business_id,
                "json_file": abs_file_path
            }
            
            print(f"Sending request to {LINGDOU_API_URL} with payload: {payload}")
            
            resp = requests.post(LINGDOU_API_URL, json=payload, timeout=60)
            
            if resp.status_code == 200:
                resp_json = resp.json()
                if resp_json.get("success"):
                    print(f"Identify success! API Response: {resp_json}")
                    # 5. Update state
                    self.state.update(synced_batch)
                    self._save_state()
                    print("State updated successfully.")
                    
                    # Cleanup temp file
                    # os.remove(temp_file) 
                else:
                    print(f"API Error: {resp_json}")
            else:
                print(f"HTTP Error: {resp.status_code} - {resp.text}")
                    
        except Exception as e:
            print(f"Push failed: {e}")

if __name__ == "__main__":
    syncer = WeChatSyncService()
    syncer.run_sync()
