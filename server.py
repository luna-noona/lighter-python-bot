import os
import time
import hmac
import inspect
from typing import Any, Dict, Literal, Optional

import lighter
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# App
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Always return JSON errors
# -----------------------------
@app.exception_handler(Exception)
async def _all_exception_handler(request: Request, exc: Exception):
    # Let FastAPI handle HTTPException normally
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": exc.__class__.__name__},
    )


# -----------------------------
# Helpers
# -----------------------------
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


def _normalise_api_key_hex(s: str) -> str:
    h = _strip_0x(s).strip()
    try:
        int(h, 16)
    except Exception:
        raise HTTPException(status_code=500, detail="LIGHTER_API_KEY_PRIVATE_KEY is not valid hex")

    if len(h) != 80:
        raise HTTPException(
            status_code=500,
            detail=f"LIGHTER_API_KEY_PRIVATE_KEY must be 80 hex chars (40 bytes). Got {len(h)}",
        )
    return h


def _parse_kv_map(raw: str) -> Dict[str, str]:
    """
    Parses "BTC-USDC=1,ETH-USDC=2" -> {"BTC-USDC":"1","ETH-USDC":"2"}
    """
    out: Dict[str, str] = {}
    raw = (raw or "").strip()
    if not raw:
        return out
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _market_index_for(market: str) -> int:
    m = _parse_kv_map(os.getenv("MARKET_INDEX_MAP", ""))
    if market not in m:
        raise HTTPException(
            status_code=400,
            detail=f"Could not resolve market_index for {market}. "
                   f"Set MARKET_INDEX_MAP like 'BTC-USDC=1,ETH-USDC=2'.",
        )
    try:
        return int(m[market])
    except Exception:
        raise HTTPException(status_code=500, detail="MARKET_INDEX_MAP has a non-integer value")


def _base_decimals_for(market: str) -> int:
    """
    BASE_DECIMALS_MAP like "BTC-USDC=8,ETH-USDC=18"
    If missing, default to 8 (common for BTC-style base units).
    """
    m = _parse_kv_map(os.getenv("BASE_DECIMALS_MAP", ""))
    if market not in m:
        return 8
    try:
        return int(m[market])
    except Exception:
        raise HTTPException(status_code=500, detail="BASE_DECIMALS_MAP has a non-integer value")


def _size_to_base_amount(market: str, size: float) -> int:
    if size <= 0:
        raise HTTPException(status_code=400, detail="size must be > 0")

    decimals = _base_decimals_for(market)
    base_amount = int(round(size * (10 ** decimals)))

    if base_amount <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"size is too small after conversion (decimals={decimals}). Try a larger size.",
        )
    return base_amount


def _auth(x_bot_token: Optional[str]) -> None:
    expected = os.getenv("BOT_TOKEN", "")
    if not expected:
        return  # no auth enabled
    if not x_bot_token:
        raise HTTPException(status_code=401, detail="Unauthorised")
    # constant-time compare
    if not hmac.compare_digest(x_bot_token, expected):
        raise HTTPException(status_code=401, detail="Unauthorised")


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


async def _maybe_await(x):
    if inspect.isawaitable(x):
        return await x
    return x


def _get_const(name: str, fallback: Optional[int] = None) -> int:
    """
    Try to pull constants from lighter.constants or lighter, else fallback.
    """
    for mod in (getattr(lighter, "constants", None), lighter):
        if mod and hasattr(mod, name):
            return int(getattr(mod, name))
    if fallback is None:
        raise HTTPException(status_code=500, detail=f"Missing SDK constant: {name}")
    return int(fallback)


async def _call_send_tx(tx_api: Any, signed: Any) -> Any:
    """
    TransactionApi.send_tx differs across SDK versions.
    We adapt:
      - send_tx(tx_type=?, tx_info=?)
      - send_tx(tx_type, tx_info)
      - send_tx(tx=?)
    """
    fn = getattr(tx_api, "send_tx", None) or getattr(tx_api, "sendTx", None)
    if not fn:
        raise HTTPException(status_code=500, detail="TransactionApi has no send_tx")

    sig = inspect.signature(fn)

    # If SDK returned already packed tx object
    if isinstance(signed, dict) and "tx_type" in signed and "tx_info" in signed:
        tx_type = signed["tx_type"]
        tx_info = signed["tx_info"]
    else:
        # some SDKs might return an object with attributes
        tx_type = getattr(signed, "tx_type", None)
        tx_info = getattr(signed, "tx_info", None)
        if tx_type is None or tx_info is None:
            raise HTTPException(
                status_code=500,
                detail="Could not extract (tx_type, tx_info) from signed order output.",
            )

    params = list(sig.parameters.keys())

    # kw style
    if "tx_type" in params and "tx_info" in params:
        return await _maybe_await(fn(tx_type=tx_type, tx_info=tx_info))

    # single 'tx' style
    if "tx" in params:
        return await _maybe_await(fn(tx=signed))

    # positional style
    return await _maybe_await(fn(tx_type, tx_info))


async def _sign_market_create_order(
    signer: Any,
    market_index: int,
    client_order_index: int,
    base_amount: int,
    is_ask: bool,
    api_key_index: int,
    nonce: Optional[int] = None,
) -> Any:
    """
    Signs a MARKET order using sign_create_order, adapting to signature differences.
    IMPORTANT: We pass market_index/base_amount ints (as per Lighter docs).
    """
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient has no sign_create_order")

    order_type_market = _get_const("ORDER_TYPE_MARKET", fallback=1)  # fallback guess if constants missing
    tif_ioc = _get_const("ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL", fallback=0)

    # For market orders, price is typically 0 in many matching engines.
    # If Lighter requires otherwise, you can change this.
    price = 0

    # Build kwargs only for params present in this SDK version
    sig = inspect.signature(fn)
    p = sig.parameters

    kwargs: Dict[str, Any] = {}

    # Common required
    if "market_index" in p:
        kwargs["market_index"] = market_index
    if "client_order_index" in p:
        kwargs["client_order_index"] = client_order_index
    if "base_amount" in p:
        kwargs["base_amount"] = base_amount
    if "price" in p:
        kwargs["price"] = price
    if "is_ask" in p:
        kwargs["is_ask"] = is_ask
    if "order_type" in p:
        kwargs["order_type"] = order_type_market
    if "time_in_force" in p:
        kwargs["time_in_force"] = tif_ioc
    if "api_key_index" in p:
        kwargs["api_key_index"] = api_key_index

    # Optional knobs
    if "reduce_only" in p:
        kwargs["reduce_only"] = False
    if "trigger_price" in p:
        kwargs["trigger_price"] = 0
    if "order_expiry" in p:
        kwargs["order_expiry"] = -1

    # Nonce: if signature expects it, include (else signer may manage internally)
    if "nonce" in p:
        kwargs["nonce"] = int(nonce) if nonce is not None else -1

    out = await _maybe_await(fn(**kwargs))

    # Some versions return (signed, err)
    if isinstance(out, (list, tuple)) and len(out) == 2:
        signed, err = out
        if err is not None:
            raise HTTPException(status_code=400, detail=f"sign_create_order failed: {err}")
        return signed

    return out


# -----------------------------
# Request model (MARKET ONLY)
# -----------------------------
class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)
    live: bool = False  # false = sign only, true = sign + send


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}


@app.get("/debug-env")
def debug_env():
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    api_key_raw = os.getenv("LIGHTER_API_KEY_PRIVATE_KEY", "")
    api_key_no0x = _strip_0x(api_key_raw).strip()

    return {
        "BASE_URL": base_url,
        "LIGHTER_ACCOUNT_INDEX": os.getenv("LIGHTER_ACCOUNT_INDEX"),
        "LIGHTER_API_KEY_INDEX": os.getenv("LIGHTER_API_KEY_INDEX"),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_raw": len(api_key_raw),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_no0x": len(api_key_no0x),
        "MARKET_INDEX_MAP_preview": os.getenv("MARKET_INDEX_MAP", "")[:120],
        "BASE_DECIMALS_MAP_preview": os.getenv("BASE_DECIMALS_MAP", "")[:120],
        "BOT_TOKEN_set": bool(os.getenv("BOT_TOKEN")),
    }


@app.post("/order")
async def place_order(req: OrderReq, x_bot_token: str | None = Header(default=None)):
    _auth(x_bot_token)

    signer = None
    api_client = None

    try:
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        market_index = _market_index_for(req.market)
        base_amount = _size_to_base_amount(req.market, req.size)

        is_ask = True if req.side == "SELL" else False
        client_order_index = int(time.time() * 1000)

        signer = make_signer_client()

        # Optional sanity check if available
        if hasattr(signer, "check_client"):
            err = signer.check_client()
            if err is not None:
                raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        # Prefer SDK wrapper if it exists and live==true
        if req.live and hasattr(signer, "create_market_order"):
            fn = getattr(signer, "create_market_order")
            sig = inspect.signature(fn)
            p = sig.parameters

            kwargs: Dict[str, Any] = {}
            if "market_index" in p:
                kwargs["market_index"] = market_index
            if "base_amount" in p:
                kwargs["base_amount"] = base_amount
            if "is_ask" in p:
                kwargs["is_ask"] = is_ask
            if "api_key_index" in p:
                kwargs["api_key_index"] = api_key_index
            if "client_order_index" in p:
                kwargs["client_order_index"] = client_order_index

            sent = await _maybe_await(fn(**kwargs))

            return {
                "success": True,
                "live": True,
                "market": req.market,
                "market_index": market_index,
                "side": req.side,
                "size": req.size,
                "base_amount": base_amount,
                "client_order_index": client_order_index,
                "response": sent,
            }

        # Otherwise: sign via sign_create_order (market), optionally send via TransactionApi
        signed = await _sign_market_create_order(
            signer=signer,
            market_index=market_index,
            client_order_index=client_order_index,
            base_amount=base_amount,
            is_ask=is_ask,
            api_key_index=api_key_index,
            nonce=None,
        )

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed market order OK (not sent). Turn live=true to broadcast.",
                "market": req.market,
                "market_index": market_index,
                "side": req.side,
                "size": req.size,
                "base_amount": base_amount,
                "client_order_index": client_order_index,
                "signed": signed,
            }

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)

        sent = await _call_send_tx(tx_api, signed)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "market_index": market_index,
            "side": req.side,
            "size": req.size,
            "base_amount": base_amount,
            "client_order_index": client_order_index,
            "response": sent,
        }

    finally:
        try:
            if signer is not None and hasattr(signer, "close"):
                await _maybe_await(signer.close())
        except Exception:
            pass
        try:
            if api_client is not None and hasattr(api_client, "close"):
                await _maybe_await(api_client.close())
        except Exception:
            pass
