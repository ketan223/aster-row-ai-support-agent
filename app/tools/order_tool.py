import os
import json
import re

# Loads mock orders database from the local orders.json file.
def _load_orders_db() -> dict:
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "orders.json")
    if not os.path.exists(db_path):
        return {"orders": [], "snapshot_at": "2026-08-15T12:00:00Z"}
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Accepts an order ID, normalizes whitespace and casing, validates the pattern,
# and returns a sanitized dictionary containing only customer-safe fields.
def lookup_order(order_id: str) -> dict:
    if not order_id:
        return {"error": "Missing order ID"}
        
    # 1. Input normalization
    # Remove leading/trailing whitespaces, replace any punctuation/dashes if required
    normalized_id = str(order_id).strip().upper()
    
    # Validation check: must match pattern ORD-\d+
    if not re.match(r"^ORD-\d+$", normalized_id):
        return {"error": "Invalid order ID format. Expected format like ORD-1007"}
        
    db = _load_orders_db()
    orders = db.get("orders", [])
    
    found_order = None
    for o in orders:
        if o.get("order_id") == normalized_id:
            found_order = o
            break
            
    if not found_order:
        return {"error": f"Order {normalized_id} not found."}
        
    # 2. Status Precedence / Stale Fields Correction
    status = found_order.get("status", "").lower()
    estimated_delivery = found_order.get("estimated_delivery")
    carrier = found_order.get("carrier")
    tracking_number = found_order.get("tracking_number")
    
    # Cancelled/Returned orders must never expose stale delivery estimates
    if status in ["cancelled", "returned"]:
        estimated_delivery = None
        
    # 3. Allowlist sanitization (Only customer-safe fields)
    sanitized_items = []
    for item in found_order.get("items", []):
        sanitized_items.append({
            "name": item.get("name"),
            "quantity": item.get("quantity"),
            "final_sale": item.get("final_sale", False)
        })
        
    sanitized_order = {
        "order_id": found_order.get("order_id"),
        "membership_tier": found_order.get("membership_tier"),
        "placed_at": found_order.get("placed_at"),
        "status": found_order.get("status"),
        "status_updated_at": found_order.get("status_updated_at"),
        "shipped_at": found_order.get("shipped_at"),
        "delivered_at": found_order.get("delivered_at"),
        "carrier": carrier,
        "tracking_number": tracking_number,
        "estimated_delivery": estimated_delivery,
        "customer_safe_message": found_order.get("customer_safe_message"),
        "items": sanitized_items
    }
    
    # Return snapshot_at info for time-related calculations
    return {
        "order": sanitized_order,
        "snapshot_at": db.get("snapshot_at", "2026-08-15T12:00:00Z")
    }
