import os
import time
import inspect
from typing import Any, Dict, Literal, Optional

import lighter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


# ---------- helpers ----------
def _need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise HTTPException(status_code=500, detail=f"Missing {name}")
    return v


def _as_int(name: str) -> int:
    try:
        return int(_need(name))
    except Exception:
        raise HTTPException(status_code=500, detail=f"{name} must be an integer")


def _strip_0x(s: str) -> str:
    return s[2:] if s.startswith("0x") else s


def _normalise_api_key_hex(s: str) -> str:
    """
    Lighter python SignerClient expects API private key as:
      40 bytes = 80 hex chars
    """
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


async def _maybe_await(x):
    if inspect.isawaitable(x):
        return await x
    return x


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


# ---------- request models ----------
class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)

    # SAFETY: live=false = sign only, live=true = actually send
    live: bool = False

    # Optional advanced knobs (leave alone for now)
    order_type: int = 0          # 0=LIMIT, 1=MARKET (depends on Lighter config; keep 0 for now)
    time_in_force: int = 0       # depends on SDK; keep 0 for now
    reduce_only: bool = False
    price: Optional[float] = None  # if None, we’ll pass 0


# ---------- endpoints ----------
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
        "LIGHTER_API_KEY_PRIVATE_KEY_len_no0x": len(api_key_no0x),
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
        "MARKET_INDEX_MAP_present": bool(os.getenv("MARKET_INDEX_MAP")),
    }


# ---------- market index resolution ----------
async def _get_market_index(order_api: Any, market_symbol: str) -> int:
    """
    Resolve market_index for a symbol like BTC-USDC.

    Priority:
    1) MARKET_INDEX_MAP env var: e.g. "BTC-USDC=1,ETH-USDC=2"
    2) Try to fetch from OrderApi if it exposes a markets/list method
    """
    # 1) env override (fastest)
    mapping = os.getenv("MARKET_INDEX_MAP", "").strip()
    if mapping:
        pairs = [p.strip() for p in mapping.split(",") if p.strip()]
        for p in pairs:
            if "=" in p:
                k, v = p.split("=", 1)
                if k.strip().upper() == market_symbol.upper():
                    return int(v.strip())

    # 2) try common SDK methods
    # We don't know exact naming, so attempt a few.
    for name in ["get_markets", "getMarkets", "markets", "list_markets", "listMarkets"]:
        fn = getattr(order_api, name, None)
        if fn:
            res = await _maybe_await(fn())
            # res may be list of objects/dicts. Try to find symbol field.
            if isinstance(res, dict) and "markets" in res:
                res = res["markets"]

            if isinstance(res, list):
                for m in res:
                    # object style
                    sym = getattr(m, "symbol", None) or getattr(m, "ticker", None) or getattr(m, "name", None)
                    idx = getattr(m, "index", None) or getattr(m, "market_index", None)
                    # dict style
                    if isinstance(m, dict):
                        sym = m.get("symbol") or m.get("ticker") or m.get("name")
                        idx = m.get("index") or m.get("market_index")

                    if sym and str(sym).upper() == market_symbol.upper():
                        if idx is None:
                            raise HTTPException(status_code=500, detail=f"Found market {market_symbol} but no index in response")
                        return int(idx)

    raise HTTPException(
        status_code=500,
        detail=(
            f"Could not resolve market_index for {market_symbol}. "
            f"Set MARKET_INDEX_MAP env var like: BTC-USDC=1,ETH-USDC=2"
        ),
    )


# ---------- nonce / sign / send (SDK version tolerant) ----------
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


async def _sign_create_order_positional(
    signer: Any,
    market_index: int,
    client_order_index: int,
    base_amount: float,
    price: float,
    is_ask: bool,
    order_type: int,
    time_in_force: int,
    reduce_only: bool,
    trigger_price: float,
    order_expiry: int,
    nonce: int,
    api_key_index: int,
):
    """
    Uses the exact positional signature you saw:

    sign_create_order(
      market_index, client_order_index, base_amount, price,
      is_ask, order_type, time_in_force, reduce_only,
      trigger_price, order_expiry=-1, nonce=-1, api_key_index=255
    )
    """
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose sign_create_order")

    # IMPORTANT: positional args to match your SDK
    out = await _maybe_await(
        fn(
            market_index,
            client_order_index,
            base_amount,
            price,
            is_ask,
            order_type,
            time_in_force,
            reduce_only,
            trigger_price,
            order_expiry,
            nonce,
            api_key_index,
        )
    )

    # Some SDKs return (signed, err)
    if isinstance(out, (list, tuple)) and len(out) == 2:
        signed, err = out
        if err is not None:
            raise HTTPException(status_code=500, detail=f"sign_create_order failed: {err}")
        return signed
    return out


async def _send_tx(tx_api: Any, tx: Any) -> Any:
    for name in ["send_tx", "sendTx"]:
        fn = getattr(tx_api, name, None)
        if fn:
            return await _maybe_await(fn(tx=tx))
    raise HTTPException(status_code=500, detail="Could not find TransactionApi send_tx method")


# ---------- main order endpoint ----------
@app.post("/order")
async def place_order(req: OrderReq):
    signer = None
    api_client = None

    try:
        # required
        _ = _need("ETH_PRIVATE_KEY")  # presence check only (your SignerClient might not need it, but keep as guard)
        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        signer = make_signer_client()
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)
        order_api = lighter.OrderApi(api_client)

        nonce = await _call_next_nonce(tx_api, account_index, api_key_index)

        market_index = await _get_market_index(order_api, req.market)

        # BUY => not ask. SELL => ask.
        is_ask = True if req.side == "SELL" else False

        client_order_index = int(time.time() * 1000)
        price = float(req.price) if req.price is not None else 0.0

        # trigger_price: for plain market/limit orders, 0.0 is fine
        trigger_price = 0.0
        order_expiry = -1

        signed_tx = await _sign_create_order_positional(
            signer=signer,
            market_index=market_index,
            client_order_index=client_order_index,
            base_amount=float(req.size),
            price=price,
            is_ask=is_ask,
            order_type=int(req.order_type),
            time_in_force=int(req.time_in_force),
            reduce_only=bool(req.reduce_only),
            trigger_price=trigger_price,
            order_expiry=order_expiry,
            nonce=int(nonce),
            api_key_index=int(api_key_index),
        )

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed OK (not sent). Set live=true to send.",
                "market": req.market,
                "market_index": market_index,
                "side": req.side,
                "is_ask": is_ask,
                "size": req.size,
                "price": price,
                "nonce": nonce,
                "client_order_index": client_order_index,
            }

        sent = await _send_tx(tx_api, signed_tx)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "market_index": market_index,
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
