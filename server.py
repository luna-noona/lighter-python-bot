import os
import time
import json
import inspect
from typing import Any, Dict, Literal, Optional, Tuple

import lighter
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

# CORS for Lovable/browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Helpers
# ---------------------------
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


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    return s


def _parse_kv_map(env_name: str) -> Dict[str, str]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return {}
    out: Dict[str, str] = {}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[_strip_quotes(k)] = _strip_quotes(v)
    return out


def _normalise_api_key_hex(s: str) -> str:
    h = _strip_0x(s).strip()
    try:
        int(h, 16)
    except Exception:
        raise HTTPException(status_code=500, detail="LIGHTER_API_KEY_PRIVATE_KEY is not valid hex")
    if len(h) != 80:
        raise HTTPException(status_code=500, detail=f"LIGHTER_API_KEY_PRIVATE_KEY must be 80 hex chars. Got {len(h)}")
    return h


async def _maybe_await(x):
    if inspect.isawaitable(x):
        return await x
    return x


# ---------------------------
# Auth (X-Bot-Token)
# ---------------------------
def _auth(x_bot_token: Optional[str]):
    expected = os.getenv("BOT_TOKEN")
    if expected and x_bot_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorised")


# ---------------------------
# Lighter clients
# ---------------------------
def make_signer_client() -> lighter.SignerClient:
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
    api_key_index = _as_int("LIGHTER_API_KEY_INDEX")
    api_private_key = _normalise_api_key_hex(_need("LIGHTER_API_KEY_PRIVATE_KEY"))

    api_private_keys = {api_key_index: api_private_key}

    return lighter.SignerClient(
        url=base_url,
        account_index=account_index,
        api_private_keys=api_private_keys,
    )


def make_api_client() -> lighter.ApiClient:
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    return lighter.ApiClient(configuration=lighter.Configuration(host=base_url))


# ---------------------------
# Request model
# ---------------------------
class OrderReq(BaseModel):
    market: str
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)
    price: Optional[float] = None  # required for this SDK build
    live: bool = False


# ---------------------------
# Market scaling / lookups
# ---------------------------
def _market_index(market: str) -> int:
    m = _parse_kv_map("MARKET_INDEX_MAP")
    if market not in m:
        raise HTTPException(status_code=400, detail=f"Unknown market {market}. Set MARKET_INDEX_MAP like BTC-USDC=1")
    return int(m[market])


def _base_decimals(market: str) -> int:
    m = _parse_kv_map("BASE_DECIMALS_MAP")
    if market not in m:
        raise HTTPException(status_code=400, detail=f"Missing BASE_DECIMALS_MAP for {market}. Example BTC-USDC=8")
    return int(m[market])


def _to_base_amount_int(market: str, size: float) -> int:
    dec = _base_decimals(market)
    return int(round(size * (10 ** dec)))


# ---------------------------
# SDK adapters
# ---------------------------
async def _call_next_nonce(tx_api: Any, account_index: int, api_key_index: int) -> int:
    for name in ["next_nonce", "nextNonce"]:
        fn = getattr(tx_api, name, None)
        if fn:
            res = await _maybe_await(fn(account_index=account_index, api_key_index=api_key_index))
            if hasattr(res, "nonce"):
                return int(res.nonce)
            if isinstance(res, dict) and "nonce" in res:
                return int(res["nonce"])
    raise HTTPException(status_code=500, detail="Could not find TransactionApi next_nonce method")


async def _sign_create_order_exact(
    signer: Any,
    market_index: int,
    client_order_index: int,
    api_key_index: int,
    base_amount_int: int,
    price: float,
    is_ask: bool,
    nonce: int,
) -> Any:
    """
    Matches the signature you previously printed:
      (market_index, client_order_index, api_key_index, base_amount, price, is_ask,
       order_type, time_in_force, reduce_only, trigger_price, order_expiry, nonce)
    """
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose sign_create_order")

    # Market-like IOC limit pattern
    order_type = 0        # LIMIT
    time_in_force = 1     # IOC
    reduce_only = False
    trigger_price = 0     # MUST be int (not float)
    order_expiry = -1     # no expiry

    args = [
        int(market_index),
        int(client_order_index),
        int(api_key_index),
        int(base_amount_int),
        float(price),
        bool(is_ask),
        int(order_type),
        int(time_in_force),
        bool(reduce_only),
        int(trigger_price),
        int(order_expiry),
        int(nonce),
    ]

    try:
        out = fn(*args)
        out = await _maybe_await(out)
        # some builds return (signed, err)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            signed, err = out
            if err is not None:
                raise Exception(str(err))
            return signed
        return out
    except Exception as e:
        sig = str(inspect.signature(fn))
        raise HTTPException(status_code=500, detail=f"sign_create_order failed. signature={sig}. error={e}")


def _deep_find(obj: Any, want_keys: Tuple[str, ...], max_depth: int = 6) -> Optional[Any]:
    if max_depth <= 0 or obj is None:
        return None

    if isinstance(obj, dict):
        for k in want_keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        for v in obj.values():
            found = _deep_find(v, want_keys, max_depth - 1)
            if found is not None:
                return found
        return None

    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = _deep_find(v, want_keys, max_depth - 1)
            if found is not None:
                return found
        return None

    for k in want_keys:
        if hasattr(obj, k):
            val = getattr(obj, k)
            if val is not None:
                return val

    if hasattr(obj, "__dict__"):
        return _deep_find(obj.__dict__, want_keys, max_depth - 1)

    return None


def _to_tx_info_string(tx_info_raw: Any) -> Optional[str]:
    if tx_info_raw is None:
        return None
    if isinstance(tx_info_raw, str):
        return tx_info_raw
    # sometimes it’s a dict/object; send_tx wants a string, so JSON it
    try:
        if isinstance(tx_info_raw, dict):
            return json.dumps(tx_info_raw, separators=(",", ":"))
        if hasattr(tx_info_raw, "__dict__"):
            return json.dumps(tx_info_raw.__dict__, separators=(",", ":"))
        return str(tx_info_raw)
    except Exception:
        return str(tx_info_raw)


def _extract_tx_info(signed: Any) -> Optional[str]:
    raw = _deep_find(signed, ("tx_info", "txInfo", "info", "tx"))
    return _to_tx_info_string(raw)


def _signed_preview(signed: Any) -> Dict[str, Any]:
    if isinstance(signed, dict):
        return {"kind": "dict", "keys": list(signed.keys())[:60]}
    if isinstance(signed, (list, tuple)):
        return {"kind": type(signed).__name__, "len": len(signed), "head_types": [type(x).__name__ for x in signed[:5]]}
    return {"kind": type(signed).__name__, "attrs": list(getattr(signed, "__dict__", {}).keys())[:60]}


async def _broadcast_send_tx(tx_api: Any, is_ask: bool, tx_info: str) -> Any:
    """
    Your SDK is telling us send_tx requires:
      send_tx(is_ask, tx_info, ...)
    and it rejects payload=/tx= kwargs.
    """
    fn = getattr(tx_api, "send_tx", None) or getattr(tx_api, "sendTx", None)
    if not fn:
        raise HTTPException(status_code=500, detail="TransactionApi does not expose send_tx")

    last_err = None

    # Try positional (most likely for your build)
    try:
        return await _maybe_await(fn(bool(is_ask), tx_info))
    except Exception as e:
        last_err = e

    # Try keyword form (some builds)
    try:
        return await _maybe_await(fn(is_ask=bool(is_ask), tx_info=tx_info))
    except Exception as e:
        last_err = e

    sig = str(inspect.signature(fn))
    raise HTTPException(status_code=500, detail=f"send_tx call failed. signature={sig}. last_error={last_err}")


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}


@app.get("/debug-env")
def debug_env():
    return {
        "BASE_URL": os.getenv("BASE_URL"),
        "LIGHTER_ACCOUNT_INDEX": os.getenv("LIGHTER_ACCOUNT_INDEX"),
        "LIGHTER_API_KEY_INDEX": os.getenv("LIGHTER_API_KEY_INDEX"),
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
        "BOT_TOKEN_set": bool(os.getenv("BOT_TOKEN")),
        "MARKET_INDEX_MAP_preview": list(_parse_kv_map("MARKET_INDEX_MAP").items())[:5],
        "BASE_DECIMALS_MAP_preview": list(_parse_kv_map("BASE_DECIMALS_MAP").items())[:5],
    }


@app.post("/order")
async def place_order(
    req: OrderReq,
    x_bot_token: Optional[str] = Header(default=None, alias="X-Bot-Token"),
):
    _auth(x_bot_token)

    signer = None
    api_client = None

    try:
        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        market_index = _market_index(req.market)
        base_amount_int = _to_base_amount_int(req.market, req.size)

        if req.price is None:
            raise HTTPException(status_code=400, detail="price is required for this SDK build")

        is_ask = True if req.side == "SELL" else False
        client_order_index = int(time.time() * 1000)

        signer = make_signer_client()
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)

        nonce = await _call_next_nonce(tx_api, account_index, api_key_index)

        signed = await _sign_create_order_exact(
            signer=signer,
            market_index=market_index,
            client_order_index=client_order_index,
            api_key_index=api_key_index,
            base_amount_int=base_amount_int,
            price=req.price,
            is_ask=is_ask,
            nonce=nonce,
        )

        tx_info = _extract_tx_info(signed)
        preview = _signed_preview(signed)

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed market-like order OK (not sent). Turn live=true to broadcast.",
                "market": req.market,
                "market_index": market_index,
                "side": req.side,
                "is_ask": is_ask,
                "size": req.size,
                "base_amount": base_amount_int,
                "price": req.price,
                "nonce": nonce,
                "client_order_index": client_order_index,
                "tx_info_present": bool(tx_info),
                "tx_info_len": len(tx_info) if tx_info else 0,
                "signed_preview": preview,
            }

        if not tx_info:
            raise HTTPException(
                status_code=500,
                detail=f"Signed output did not include tx_info, cannot broadcast. signed_preview={preview}",
            )

        sent = await _broadcast_send_tx(tx_api, is_ask=is_ask, tx_info=tx_info)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "market_index": market_index,
            "side": req.side,
            "is_ask": is_ask,
            "size": req.size,
            "base_amount": base_amount_int,
            "price": req.price,
            "nonce": nonce,
            "client_order_index": client_order_index,
            "tx_info_present": True,
            "tx_info_len": len(tx_info),
            "signed_preview": preview,
            "response": sent,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if signer is not None:
                await signer.close()
        except Exception:
            pass
        try:
            if api_client is not None:
                await api_client.close()
        except Exception:
            pass
