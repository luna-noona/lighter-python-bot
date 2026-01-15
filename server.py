import os
import time
import inspect
from typing import Any, Dict, Literal, Optional, Tuple, Union

import lighter
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

# Allow Loveable/Browser calls (tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Helpers / env parsing
# -----------------------
def _need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise HTTPException(status_code=500, detail=f"Missing env var: {name}")
    return v


def _as_int(name: str) -> int:
    try:
        return int(_need(name))
    except Exception:
        raise HTTPException(status_code=500, detail=f"{name} must be an integer")


def _strip_0x(s: str) -> str:
    return s[2:] if s.startswith("0x") else s


def _parse_map(env_name: str) -> Dict[str, str]:
    """
    Parses env like:  BTC-USDC=1,ETH-USDC=2
    Returns dict[str,str]
    """
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise HTTPException(status_code=500, detail=f"{env_name} malformed near: {part}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_int_map(env_name: str) -> Dict[str, int]:
    m = _parse_map(env_name)
    out: Dict[str, int] = {}
    for k, v in m.items():
        try:
            out[k] = int(v)
        except Exception:
            raise HTTPException(status_code=500, detail=f"{env_name} value for {k} must be int")
    return out


def _normalise_api_key_hex(s: str) -> str:
    """
    Lighter SignerClient expects API private key as 40 bytes = 80 hex chars.
    """
    h = _strip_0x(s).strip()
    try:
        int(h, 16)
    except Exception:
        raise HTTPException(status_code=500, detail="LIGHTER_API_KEY_PRIVATE_KEY is not valid hex")
    if len(h) != 80:
        raise HTTPException(
            status_code=500,
            detail=f"LIGHTER_API_KEY_PRIVATE_KEY must be 80 hex chars. Got {len(h)}",
        )
    return h


async def _maybe_await(x):
    if inspect.isawaitable(x):
        return await x
    return x


# -----------------------
# Auth
# -----------------------
def _auth(x_bot_token: Optional[str]) -> None:
    expected = os.getenv("BOT_TOKEN")
    # If BOT_TOKEN is set, require it
    if expected and x_bot_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorised")


# -----------------------
# Clients
# -----------------------
def make_signer_client() -> lighter.SignerClient:
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
    api_key_index = _as_int("LIGHTER_API_KEY_INDEX")
    api_private_key = _normalise_api_key_hex(_need("LIGHTER_API_KEY_PRIVATE_KEY"))
    api_private_keys = {api_key_index: api_private_key}
    return lighter.SignerClient(url=base_url, account_index=account_index, api_private_keys=api_private_keys)


def make_api_client() -> lighter.ApiClient:
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    return lighter.ApiClient(configuration=lighter.Configuration(host=base_url))


# -----------------------
# Config maps from env
# -----------------------
MARKET_INDEX_MAP = _parse_int_map("MARKET_INDEX_MAP")   # BTC-USDC=1,...
BASE_DECIMALS_MAP = _parse_int_map("BASE_DECIMALS_MAP") # BTC-USDC=8,...
AMOUNT_SCALE_MAP = _parse_int_map("AMOUNT_SCALE_MAP")   # BTC-USDC=100000000,...


def _market_index(market: str) -> int:
    if market not in MARKET_INDEX_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown market '{market}'. Set MARKET_INDEX_MAP env var.")
    return MARKET_INDEX_MAP[market]


def _base_decimals(market: str) -> int:
    if market not in BASE_DECIMALS_MAP:
        raise HTTPException(status_code=400, detail=f"Missing base decimals for '{market}'. Set BASE_DECIMALS_MAP.")
    return BASE_DECIMALS_MAP[market]


def _amount_scale(market: str) -> int:
    # Prefer explicit AMOUNT_SCALE_MAP if provided (you have it for BTC-USDC)
    if market in AMOUNT_SCALE_MAP:
        return AMOUNT_SCALE_MAP[market]
    # Fallback to 10**base_decimals
    return 10 ** _base_decimals(market)


def _to_base_amount_int(market: str, size: float) -> int:
    scale = _amount_scale(market)
    amt = int(round(size * scale))
    if amt <= 0:
        raise HTTPException(status_code=400, detail="size too small after scaling")
    return amt


def _to_order_price_int(price: float) -> int:
    """
    IMPORTANT:
    Your earlier 'OrderPrice < 1' came from price scaling going to 0.
    In the Lighter SDK, 'order_price' is an integer.
    For *market orders* we still need a non-zero cap.
    Here we treat the 'price' you send as already in the units Lighter expects (human-ish),
    and just enforce >= 1 once int-cast.
    """
    p = int(round(price))
    if p < 1:
        raise HTTPException(status_code=400, detail="price too low; must be >= 1")
    return p


# -----------------------
# Request models
# -----------------------
class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)
    # For market orders, this is the PRICE CAP:
    # BUY: max you’re willing to pay
    # SELL: min you’re willing to accept
    price: float = Field(..., gt=0)
    live: bool = False


# -----------------------
# SDK adapters
# -----------------------
async def _call_next_nonce(tx_api: Any, account_index: int, api_key_index: int) -> int:
    for name in ["next_nonce", "nextNonce"]:
        fn = getattr(tx_api, name, None)
        if fn:
            res = await _maybe_await(fn(account_index=account_index, api_key_index=api_key_index))
            # res might be obj or dict
            if hasattr(res, "nonce"):
                return int(res.nonce)
            if isinstance(res, dict) and "nonce" in res:
                return int(res["nonce"])
    # Some SDKs just let you omit nonce; return -1 means "auto"
    return -1


def _extract_tx_payload(signed: Any) -> Tuple[Optional[int], Optional[str], Any]:
    """
    Try to normalise "signed order" result into (tx_type, tx_info, raw)
    Many lighter-sdk builds return:
      - tuple(tx_type, tx_info)
      - dict with tx_type/tx_info
      - object with tx_type/tx_info
      - sometimes just tx_info
    """
    tx_type = None
    tx_info = None

    if isinstance(signed, (list, tuple)) and len(signed) == 2:
        tx_type, tx_info = signed[0], signed[1]
        return tx_type, tx_info, signed

    if isinstance(signed, dict):
        tx_type = signed.get("tx_type") or signed.get("txType")
        tx_info = signed.get("tx_info") or signed.get("txInfo") or signed.get("tx")
        return tx_type, tx_info, signed

    # object-like
    tx_type = getattr(signed, "tx_type", None) or getattr(signed, "txType", None)
    tx_info = getattr(signed, "tx_info", None) or getattr(signed, "txInfo", None) or getattr(signed, "tx", None)
    return tx_type, tx_info, signed


async def _broadcast(
    signer: Any,
    tx_api: Any,
    signed: Any,
) -> Any:
    """
    Best-effort broadcast across SDK variants.
    """
    tx_type, tx_info, raw = _extract_tx_payload(signed)

    # 1) Prefer SignerClient.send_tx(tx_type, tx_info)
    send_fn = getattr(signer, "send_tx", None) or getattr(signer, "sendTx", None)
    if send_fn:
        # Try (tx_type, tx_info) then just (tx_info)
        try:
            if tx_type is not None and tx_info is not None:
                return await _maybe_await(send_fn(tx_type, tx_info))
        except Exception:
            pass
        try:
            if tx_info is not None:
                return await _maybe_await(send_fn(tx_info))
        except Exception:
            pass

    # 2) Try TransactionApi.send_tx / sendTx
    tx_send = getattr(tx_api, "send_tx", None) or getattr(tx_api, "sendTx", None)
    if tx_send:
        # Some accept named args, some positional
        try:
            if tx_type is not None and tx_info is not None:
                return await _maybe_await(tx_send(tx_type=tx_type, tx_info=tx_info))
        except Exception:
            pass
        try:
            if tx_type is not None and tx_info is not None:
                return await _maybe_await(tx_send(tx_type, tx_info))
        except Exception:
            pass
        try:
            # Some accept a single "tx" object
            return await _maybe_await(tx_send(tx=raw))
        except Exception:
            pass
        try:
            return await _maybe_await(tx_send(raw))
        except Exception:
            pass

    raise HTTPException(status_code=500, detail="Could not broadcast tx (send method not compatible with this lighter-sdk build).")


async def _sign_market_order(
    signer: Any,
    *,
    market: str,
    side: str,
    size: float,
    price_cap: float,
    nonce: int,
    api_key_index: int,
) -> Any:
    """
    MARKET orders only.

    Your lighter-sdk signature (from your logs) looks like:
      sign_create_order(
        market_index, client_order_index, base_amount, price,
        is_ask, order_type, time_in_force, reduce_only=False,
        trigger_price=0, order_expiry=-1, nonce=-1, api_key_index=255
      )

    We call POSITIONALLY to avoid keyword-arg incompatibilities.
    """
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient has no sign_create_order")

    market_index = _market_index(market)
    base_amount = _to_base_amount_int(market, size)
    order_price = _to_order_price_int(price_cap)

    is_ask = True if side == "SELL" else False

    # These values worked in your earlier successful runs:
    # order_type=1 (market), time_in_force=0, order_expiry=-1
    order_type = 1              # MARKET
    time_in_force = 0
    reduce_only = False
    trigger_price = 0
    order_expiry = -1

    client_order_index = int(time.time() * 1000)

    # Build positional args to match the signature shape you saw
    args = [
        market_index,
        client_order_index,
        base_amount,
        order_price,
        is_ask,
        order_type,
        time_in_force,
        reduce_only,
        trigger_price,
        order_expiry,
        nonce,
        api_key_index,
    ]

    # Some SDKs may return (result, err)
    out = await _maybe_await(fn(*args))
    if isinstance(out, (list, tuple)) and len(out) == 2:
        signed, err = out
        if err is not None:
            raise HTTPException(status_code=500, detail=f"sign_create_order failed: {err}")
        return signed

    return out


# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}


@app.get("/debug-env")
def debug_env():
    # DO NOT print secrets; only show presence/length
    api_key_raw = os.getenv("LIGHTER_API_KEY_PRIVATE_KEY", "")
    return {
        "BASE_URL": os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai"),
        "LIGHTER_ACCOUNT_INDEX": os.getenv("LIGHTER_ACCOUNT_INDEX"),
        "LIGHTER_API_KEY_INDEX": os.getenv("LIGHTER_API_KEY_INDEX"),
        "LIGHTER_API_KEY_PRIVATE_KEY_len": len(_strip_0x(api_key_raw).strip()),
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
        "BOT_TOKEN_set": bool(os.getenv("BOT_TOKEN")),
        "MARKET_INDEX_MAP_preview": ",".join(list(MARKET_INDEX_MAP.keys())[:5]),
        "BASE_DECIMALS_MAP_preview": ",".join([f"{k}={v}" for k, v in list(BASE_DECIMALS_MAP.items())[:5]]),
        "AMOUNT_SCALE_MAP_preview": ",".join([f"{k}={v}" for k, v in list(AMOUNT_SCALE_MAP.items())[:5]]),
    }


@app.post("/order")
async def place_order(req: OrderReq, x_bot_token: Optional[str] = Header(default=None, alias="X-Bot-Token")):
    _auth(x_bot_token)

    signer = None
    api_client = None

    try:
        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        signer = make_signer_client()
        check = signer.check_client()
        if check is not None:
            raise HTTPException(status_code=500, detail=f"Signer check_client failed: {check}")

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)

        nonce = await _call_next_nonce(tx_api, account_index, api_key_index)

        signed = await _sign_market_order(
            signer,
            market=req.market,
            side=req.side,
            size=req.size,
            price_cap=req.price,
            nonce=nonce,
            api_key_index=api_key_index,
        )

        # Always return a useful payload
        tx_type, tx_info, raw = _extract_tx_payload(signed)

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed market order OK (not sent). Set live=true to broadcast.",
                "market": req.market,
                "market_index": _market_index(req.market),
                "side": req.side,
                "size": req.size,
                "base_amount": _to_base_amount_int(req.market, req.size),
                "price": req.price,
                "order_price_int": _to_order_price_int(req.price),
                "nonce": nonce,
                "client_order_index": int(time.time() * 1000),
                "tx_type": tx_type,
                "tx_info_present": tx_info is not None,
            }

        # LIVE broadcast
        resp = await _broadcast(signer, tx_api, signed)

        return {
            "success": True,
            "live": True,
            "message": "Broadcast attempted.",
            "market": req.market,
            "market_index": _market_index(req.market),
            "side": req.side,
            "size": req.size,
            "base_amount": _to_base_amount_int(req.market, req.size),
            "price": req.price,
            "order_price_int": _to_order_price_int(req.price),
            "nonce": nonce,
            "tx_type": tx_type,
            "response": resp,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if signer is not None:
                await _maybe_await(signer.close())
        except Exception:
            pass
        try:
            if api_client is not None:
                await _maybe_await(api_client.close())
        except Exception:
            pass
