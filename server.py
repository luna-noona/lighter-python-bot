import os
import time
from typing import Any, Dict, Literal, Optional, Tuple

import httpx
import zklighter
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

# CORS (Loveable runs in-browser, so allow it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Helpers
# ----------------------------
def _need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise HTTPException(status_code=500, detail=f"Missing env var: {name}")
    return v


def _auth(x_bot_token: Optional[str]) -> None:
    expected = os.getenv("BOT_TOKEN")
    # If BOT_TOKEN is set, require it.
    if expected and x_bot_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorised")


def _parse_map_env(name: str) -> Dict[str, str]:
    """
    Parses env like:
      BTC-USDC=1,ETH-USDC=2
    into dict.
    """
    raw = os.getenv(name, "").strip()
    out: Dict[str, str] = {}
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _market_index(market: str) -> int:
    m = _parse_map_env("MARKET_INDEX_MAP")
    if market not in m:
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve market_index for {market}. Set MARKET_INDEX_MAP env var like BTC-USDC=1,ETH-USDC=2",
        )
    try:
        return int(m[market])
    except Exception:
        raise HTTPException(status_code=500, detail=f"MARKET_INDEX_MAP value for {market} must be int")


def _base_decimals(market: str) -> int:
    m = _parse_map_env("BASE_DECIMALS_MAP")
    if market not in m:
        raise HTTPException(
            status_code=500,
            detail=f"Missing BASE_DECIMALS_MAP for {market}. Example: BTC-USDC=8,ETH-USDC=18",
        )
    try:
        return int(m[market])
    except Exception:
        raise HTTPException(status_code=500, detail=f"BASE_DECIMALS_MAP value for {market} must be int")


def _price_decimals(market: str) -> int:
    # Optional. If not set, default to 2 (matches most examples).
    m = _parse_map_env("PRICE_DECIMALS_MAP")
    if market in m:
        try:
            return int(m[market])
        except Exception:
            raise HTTPException(status_code=500, detail=f"PRICE_DECIMALS_MAP value for {market} must be int")
    return 2


def _size_to_base_amount(market: str, size_float: float) -> int:
    # Convert to int in base units using BASE_DECIMALS_MAP
    dec = _base_decimals(market)
    scale = 10 ** dec
    amt = int(round(size_float * scale))
    if amt <= 0:
        raise HTTPException(status_code=400, detail="Size too small after decimal conversion")
    return amt


def _price_to_int(market: str, price_float: float) -> int:
    dec = _price_decimals(market)
    scale = 10 ** dec
    p = int(round(price_float * scale))
    if p < 1:
        p = 1
    return p


async def _get_best_prices(base_url: str, market_index: int) -> Tuple[float, float]:
    """
    Returns (best_bid, best_ask) as floats from order book.
    """
    client = zklighter.ApiClient(configuration=zklighter.Configuration(host=base_url))
    try:
        order_api = zklighter.OrderApi(client)
        ob = await order_api.order_book_orders(market_id=market_index, limit=10)
        best_bid = float(ob.bids[0].price) if ob.bids else 0.0
        best_ask = float(ob.asks[0].price) if ob.asks else 0.0
        if best_bid <= 0 or best_ask <= 0:
            raise HTTPException(status_code=500, detail="Could not read best bid/ask from order book")
        return best_bid, best_ask
    finally:
        try:
            await client.close()
        except Exception:
            pass


def _extract_tx_payload(tx_obj: Any) -> Tuple[Optional[int], Optional[Any]]:
    """
    We want something we can POST to /api/v1/sendTx:
      {"tx_type": <int>, "tx_info": <object>}
    Different SDK builds represent tx differently, so we try a few shapes.
    """
    # 1) direct attrs
    tx_type = getattr(tx_obj, "tx_type", None)
    tx_info = getattr(tx_obj, "tx_info", None)
    if tx_type is not None and tx_info is not None:
        return int(tx_type), tx_info

    # 2) dict-like
    if isinstance(tx_obj, dict):
        if "tx_type" in tx_obj and "tx_info" in tx_obj:
            return int(tx_obj["tx_type"]), tx_obj["tx_info"]

    # 3) to_json()
    to_json = getattr(tx_obj, "to_json", None)
    if callable(to_json):
        j = to_json()
        if isinstance(j, dict) and "tx_type" in j and "tx_info" in j:
            return int(j["tx_type"]), j["tx_info"]

    # 4) model_dump()
    model_dump = getattr(tx_obj, "model_dump", None)
    if callable(model_dump):
        j = model_dump()
        if isinstance(j, dict) and "tx_type" in j and "tx_info" in j:
            return int(j["tx_type"]), j["tx_info"]

    return None, None


async def _broadcast_sendtx_rest(base_url: str, tx_type: int, tx_info: Any) -> Any:
    """
    REST broadcast:
      POST {BASE_URL}/api/v1/sendTx
      body: {"tx_type": <int>, "tx_info": <...>}
    """
    url = base_url.rstrip("/") + "/api/v1/sendTx"
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.post(url, json={"tx_type": tx_type, "tx_info": tx_info})
        # Lighter often returns JSON either way; keep raw if not JSON.
        try:
            data = r.json()
        except Exception:
            data = {"status_code": r.status_code, "text": r.text}
        return data


# ----------------------------
# Request model
# ----------------------------
class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0, description="Base size (e.g. BTC amount)")
    live: bool = False

    # Optional slippage guard (for market-like execution)
    slippage_bps: int = Field(default=50, ge=0, le=2000, description="0.01% = 1 bps. Default 50 bps (0.50%).")


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}


@app.get("/debug-env")
def debug_env():
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    return {
        "BASE_URL": base_url,
        "LIGHTER_ACCOUNT_INDEX": os.getenv("LIGHTER_ACCOUNT_INDEX"),
        "LIGHTER_API_KEY_INDEX": os.getenv("LIGHTER_API_KEY_INDEX"),
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
        "BOT_TOKEN_set": bool(os.getenv("BOT_TOKEN")),
        "MARKET_INDEX_MAP_preview": os.getenv("MARKET_INDEX_MAP", ""),
        "BASE_DECIMALS_MAP_preview": os.getenv("BASE_DECIMALS_MAP", ""),
        "PRICE_DECIMALS_MAP_preview": os.getenv("PRICE_DECIMALS_MAP", ""),
    }


@app.post("/order")
async def place_order(req: OrderReq, x_bot_token: Optional[str] = Header(default=None)):
    _auth(x_bot_token)

    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    eth_private_key = _need("ETH_PRIVATE_KEY")

    market_index = _market_index(req.market)
    base_amount = _size_to_base_amount(req.market, req.size)

    # MARKET order needs a price guard in many matching engines (slippage bound).
    best_bid, best_ask = await _get_best_prices(base_url, market_index)

    if req.side == "BUY":
        ref_price = best_ask
        # allow paying up to +slippage
        bounded = ref_price * (1.0 + (req.slippage_bps / 10_000.0))
    else:
        ref_price = best_bid
        # allow selling down to -slippage
        bounded = ref_price * (1.0 - (req.slippage_bps / 10_000.0))

    price_int = _price_to_int(req.market, bounded)

    # order_expiry: give it ~5 minutes
    order_expiry = int((time.time() + 5 * 60) * 1000)
    client_order_index = int(time.time() * 1000)

    client = None
    try:
        # NOTE: Official docs show SignerClient(BASE_URL, PRIVATE_KEY)
        # (and create_order rather than sign_create_order).
        client = zklighter.SignerClient(base_url, eth_private_key)

        tx = await client.create_order(
            market_index=market_index,
            client_order_index=client_order_index,
            base_amount=base_amount,
            price=price_int,
            is_ask=0 if req.side == "BUY" else 1,
            order_type=zklighter.SignerClient.ORDER_TYPE_MARKET,
            time_in_force=zklighter.SignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
            order_expiry=order_expiry,
        )

        # Some SDK builds return (tx, response). Handle both.
        response_obj = None
        tx_obj = tx
        if isinstance(tx, (list, tuple)) and len(tx) == 2:
            tx_obj, response_obj = tx[0], tx[1]

        tx_type, tx_info = _extract_tx_payload(tx_obj)

        result = {
            "success": True,
            "live": req.live,
            "market": req.market,
            "market_index": market_index,
            "side": req.side,
            "size": req.size,
            "base_amount": base_amount,
            "slippage_bps": req.slippage_bps,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "price_bound_used": bounded,
            "price_int": price_int,
            "order_expiry": order_expiry,
            "client_order_index": client_order_index,
            "sdk_response": getattr(response_obj, "model_dump", lambda: response_obj)() if response_obj is not None else None,
            "tx_type": tx_type,
            "tx_info_present": tx_info is not None,
        }

        if not req.live:
            # Return signed payload details (useful for debugging), but don't broadcast.
            return result

        # LIVE: broadcast via REST
        if tx_type is None or tx_info is None:
            raise HTTPException(
                status_code=500,
                detail="Signed output did not include tx_type/tx_info (cannot broadcast). Check zklighter SDK version.",
            )

        sent = await _broadcast_sendtx_rest(base_url, tx_type, tx_info)
        result["broadcast"] = sent
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if client is not None:
                await client.close()
        except Exception:
            pass
