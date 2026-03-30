import json
import os
import time
import requests
import argparse
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
# OLD STATE_FILE dependency removed
LINGDOU_API_URL = "http://localhost:8008/api/process_data"  # Default local API

class WeChatSyncService:
    def __init__(self, app_id: str = None, business_id: str = None):
        self.config = self._load_config()
        
        # 1. 优先使用命令行传参，没有则从遗留 config 里靠正则/切片提取
        self.app_id = app_id
        if not self.app_id:
            token_url = self.config.get("token_url", "")
            if "wx_store/" in token_url:
                self.app_id = token_url.split("wx_store/")[1].split("/")[0]
            else:
                self.app_id = "default_store"
                
        self.business_id = business_id or self.config.get("business_id", "wechat_shop")
        
        # 2. 【多租户核心】每个小店必须拥有独立的状态记录文件！
        self.state_file = os.path.join(file_dir, f"synced_ids_{self.app_id}.json")
        self.state = self._load_state()
        
        # 3. 动态拼装内部 token
        dynamic_token_url = f"http://47.100.14.93/backend//lingdou/wx_store/{self.app_id}/token?secret=1024@Yinyu"
        
        # Initialize connector
        self.connector = WeChatShopConnector(
            token_url=dynamic_token_url
        )

    def _load_config(self) -> Dict:
        if not os.path.exists(CONFIG_FILE):
             # Fallback or error
             print("Config file not found!")
             return {}
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_state(self) -> Set[str]:
        if not os.path.exists(self.state_file):
            return set()
        with open(self.state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get("synced_ids", []))

    def _save_state(self):
        with open(self.state_file, 'w', encoding='utf-8') as f:
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
            # 适配通用知识库配置字段 (Fallback)
            # "title": raw_product.get("title"),
            # "content": desc_text,
            # 原本的中文后备字段
            # "商品名": raw_product.get("title"),
            # "价格": price,
            # "参数": attrs
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
    parser = argparse.ArgumentParser(description="Multi-tenant WeChat Shop Sync Service")
    parser.add_argument("--app_id", type=str, help="WeChat Shop AppID (e.g. wxad76052896802106)")
    parser.add_argument("--business_id", type=str, help="LingDou Knowledge Base ID (e.g. wechat_shop)")
    
    args = parser.parse_args()
    
    syncer = WeChatSyncService(app_id=args.app_id, business_id=args.business_id)
    syncer.run_sync()
