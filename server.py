import os
import time
import inspect
from typing import Any, Dict, Literal, Optional, Tuple

import lighter
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

# CORS for Lovable / browser calls (tighten later)
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
    """
    Parses env like: BTC-USDC=1,ETH-USDC=2
    Also tolerates accidental quotes: "BTC-USDC"="1"
    """
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return {}
    out: Dict[str, str] = {}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = _strip_quotes(k)
        v = _strip_quotes(v)
        out[k] = v
    return out


def _normalise_api_key_hex(s: str) -> str:
    """
    SignerClient expects API private key as 40 bytes = 80 hex chars (no 0x).
    """
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
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)

    # This SDK build still requires a price field (even for "market-like" IOC behaviour)
    price: Optional[float] = None

    live: bool = False  # live=false => sign only; live=true => broadcast


# ---------------------------
# Market scaling / lookups
# ---------------------------
def _market_index(market: str) -> int:
    m = _parse_kv_map("MARKET_INDEX_MAP")
    if market not in m:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown market {market}. Set MARKET_INDEX_MAP like: BTC-USDC=1,ETH-USDC=2",
        )
    return int(m[market])


def _base_decimals(market: str) -> int:
    m = _parse_kv_map("BASE_DECIMALS_MAP")
    if market not in m:
        raise HTTPException(
            status_code=400,
            detail=f"Missing BASE_DECIMALS_MAP for {market}. Example: BTC-USDC=8,ETH-USDC=18",
        )
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


def _extract_tx_type_info(signed: Any) -> Tuple[Optional[int], Optional[str]]:
    tx_type = None
    tx_info = None

    if isinstance(signed, dict):
        tx_type = signed.get("tx_type") or signed.get("txType")
        tx_info = signed.get("tx_info") or signed.get("txInfo")
        if tx_type is not None:
            try:
                tx_type = int(tx_type)
            except Exception:
                pass
        if tx_info is not None:
            tx_info = str(tx_info)
        return tx_type, tx_info

    for a in ["tx_type", "txType"]:
        if hasattr(signed, a):
            try:
                tx_type = int(getattr(signed, a))
            except Exception:
                pass
    for a in ["tx_info", "txInfo"]:
        if hasattr(signed, a):
            try:
                tx_info = str(getattr(signed, a))
            except Exception:
                pass

    if isinstance(signed, (list, tuple)) and len(signed) >= 2:
        try:
            tx_type = int(signed[0])
        except Exception:
            pass
        try:
            tx_info = str(signed[1])
        except Exception:
            pass

    return tx_type, tx_info


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
    Uses the signature shape your lighter-sdk build is showing in errors.
    IMPORTANT FIX: trigger_price must be int (0), not float (0.0).
    """
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose sign_create_order")

    # "Market-like" behaviour via IOC limit (common pattern)
    order_type = 0        # LIMIT
    time_in_force = 1     # IOC (typical)
    reduce_only = False
    trigger_price = 0     # MUST be int for your build
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
        int(trigger_price),     # <--- FIXED
        int(order_expiry),
        int(nonce),
    ]

    try:
        out = fn(*args)
        out = await _maybe_await(out)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            signed, err = out
            if err is not None:
                raise Exception(str(err))
            return signed
        return out
    except Exception as e:
        sig = str(inspect.signature(fn))
        raise HTTPException(
            status_code=500,
            detail=f"sign_create_order failed. signature={sig}. error={e}",
        )


async def _broadcast_tx(tx_api: Any, tx_type: int, tx_info: str) -> Any:
    """
    Broadcast with multiple possible send signatures across lighter-sdk builds.
    """
    candidates = []
    for name in ["send_tx", "sendTx", "broadcast_tx", "broadcastTx"]:
        fn = getattr(tx_api, name, None)
        if fn:
            candidates.append((name, fn))

    if not candidates:
        raise HTTPException(status_code=500, detail="No send/broadcast method found on TransactionApi")

    last_err = None
    for name, fn in candidates:
        # keyword style
        try:
            return await _maybe_await(fn(tx_type=tx_type, tx_info=tx_info))
        except Exception as e:
            last_err = e

        # positional style
        try:
            return await _maybe_await(fn(tx_type, tx_info))
        except Exception as e:
            last_err = e

        # dict style
        try:
            return await _maybe_await(fn({"tx_type": tx_type, "tx_info": tx_info}))
        except Exception as e:
            last_err = e

        # tx=... style
        try:
            ReqSendTx = getattr(lighter, "ReqSendTx", None)
            if ReqSendTx:
                req = ReqSendTx(tx_type=tx_type, tx_info=tx_info)
                return await _maybe_await(fn(tx=req))
        except Exception as e:
            last_err = e

        try:
            return await _maybe_await(fn(tx={"tx_type": tx_type, "tx_info": tx_info}))
        except Exception as e:
            last_err = e

    sigs = [f"{n}{inspect.signature(f)}" for (n, f) in candidates]
    raise HTTPException(
        status_code=500,
        detail=f"Could not broadcast tx (send method not compatible with this lighter-sdk build). "
               f"methods_tried={sigs}. last_error={last_err}",
    )


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}


@app.get("/debug-env")
def debug_env():
    base_url = os.getenv("BASE_URL", "")
    market_index_map = _parse_kv_map("MARKET_INDEX_MAP")
    base_decimals_map = _parse_kv_map("BASE_DECIMALS_MAP")
    amount_scale_map = _parse_kv_map("AMOUNT_SCALE_MAP")

    api_key_raw = os.getenv("LIGHTER_API_KEY_PRIVATE_KEY", "")
    api_key_no0x = _strip_0x(api_key_raw).strip()

    return {
        "BASE_URL": base_url,
        "LIGHTER_ACCOUNT_INDEX": os.getenv("LIGHTER_ACCOUNT_INDEX"),
        "LIGHTER_API_KEY_INDEX": os.getenv("LIGHTER_API_KEY_INDEX"),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_raw": len(api_key_raw),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_no0x": len(api_key_no0x),
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
        "BOT_TOKEN_set": bool(os.getenv("BOT_TOKEN")),
        "MARKET_INDEX_MAP_preview": list(market_index_map.items())[:5],
        "BASE_DECIMALS_MAP_preview": list(base_decimals_map.items())[:5],
        "AMOUNT_SCALE_MAP_preview": list(amount_scale_map.items())[:5],
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
            raise HTTPException(status_code=400, detail="price is required for this build")

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

        tx_type, tx_info = _extract_tx_type_info(signed)

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed market-like order OK (not sent). Turn live=true to broadcast.",
                "market": req.market,
                "market_index": market_index,
                "side": req.side,
                "size": req.size,
                "base_amount": base_amount_int,
                "price": req.price,
                "nonce": nonce,
                "client_order_index": client_order_index,
                "tx_type": tx_type,
                "tx_info_present": bool(tx_info),
            }

        if tx_type is None or not tx_info:
            raise HTTPException(
                status_code=500,
                detail="Signed output did not include tx_type/tx_info, cannot broadcast.",
            )

        sent = await _broadcast_tx(tx_api, tx_type=tx_type, tx_info=tx_info)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "market_index": market_index,
            "side": req.side,
            "size": req.size,
            "base_amount": base_amount_int,
            "price": req.price,
            "nonce": nonce,
            "client_order_index": client_order_index,
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
