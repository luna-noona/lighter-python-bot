import os
import time
import inspect
from typing import Any, Dict, Literal, Optional

import lighter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

# -----------------------------
# helpers
# -----------------------------
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
    Lighter python SignerClient expects the API private key as 40 bytes = 80 hex chars.
    - Accept optional 0x prefix and remove it.
    - Must be valid hex.
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
    """Await if x is awaitable, else return x."""
    if inspect.isawaitable(x):
        return await x
    return x


# -----------------------------
# clients
# -----------------------------
def make_signer_client() -> lighter.SignerClient:
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
    api_key_index = _as_int("LIGHTER_API_KEY_INDEX")
    api_private_key = _normalise_api_key_hex(_need("LIGHTER_API_KEY_PRIVATE_KEY"))

    # SignerClient wants Dict[int, str]
    api_private_keys = {api_key_index: api_private_key}

    return lighter.SignerClient(
        url=base_url,
        account_index=account_index,
        api_private_keys=api_private_keys,
    )


def make_api_client() -> lighter.ApiClient:
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    return lighter.ApiClient(configuration=lighter.Configuration(host=base_url))


# -----------------------------
# request models
# -----------------------------
class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)

    # live=false => sign only (safe)
    # live=true  => send (real)
    live: bool = False

    # Optional override if your SDK requires it
    order_type: Optional[int] = None


# -----------------------------
# endpoints
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
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
        "MARKET_INDEX_MAP_present": bool(os.getenv("MARKET_INDEX_MAP")),
    }


# -----------------------------
# market mapping
# -----------------------------
def _market_index_for(market: str) -> int:
    """
    Lighter SDK method you’re on expects market_index not a ticker string.
    Provide MARKET_INDEX_MAP env var like:
      BTC-USDC=1,ETH-USDC=2
    """
    mapping_raw = os.getenv("MARKET_INDEX_MAP", "").strip()
    if not mapping_raw:
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve market_index for {market}. Set MARKET_INDEX_MAP env var like: BTC-USDC=1,ETH-USDC=2",
        )

    mapping: Dict[str, int] = {}
    for part in mapping_raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            continue
        try:
            mapping[k] = int(v)
        except Exception:
            continue

    if market not in mapping:
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve market_index for {market}. MARKET_INDEX_MAP missing {market}. Example: BTC-USDC=1,ETH-USDC=2",
        )
    return mapping[market]


# -----------------------------
# sdk adapters (handle version differences)
# -----------------------------
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


def _int_if_close(x: float) -> int:
    """
    Some SDK builds want base_amount as int (e.g., in base units).
    If we can’t infer decimals safely, we’ll default to int(size * 1e8) for BTC-like.
    You can tweak BASE_AMOUNT_MULT env var if needed.
    """
    mult = int(os.getenv("BASE_AMOUNT_MULT", "100000000"))  # 1e8 default
    return int(round(x * mult))


async def _sign_create_order(
    signer: Any,
    eth_private_key: str,
    market_index: int,
    side: int,
    base_amount: float,
    nonce: int,
    client_order_index: int,
    api_key_index: int,
    order_type: Optional[int],
):
    """
    Your SDK signature (from your error) looks like:
      sign_create_order(
        market_index, client_order_index, base_amount, price, is_ask,
        order_type, time_in_force, reduce_only, trigger_price,
        order_expiry, nonce, api_key_index
      )

    So we build those args positionally.
    We also try a few permutations because versions differ slightly.
    """
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose sign_create_order")

    # Market order: use price=0, trigger_price=0, expiry=-1, tif=0, reduce_only=0
    price = 0
    trigger_price = 0
    time_in_force = 0
    reduce_only = 0
    order_expiry = -1

    # is_ask: BUY => False, SELL => True
    is_ask = True if side == 1 else False

    # order_type: default 1 (you already discovered this fixes the float/int error path)
    ot = int(order_type) if order_type is not None else int(os.getenv("DEFAULT_ORDER_TYPE", "1"))

    # base_amount: some versions want int
    ba_int = _int_if_close(base_amount)

    last_err = None

    # Most likely positional layouts we see in wild:
    candidates = [
        # (market_index, client_order_index, base_amount_int, price, is_ask, order_type, tif, reduce_only, trigger_price, expiry, nonce, api_key_index)
        (market_index, client_order_index, ba_int, price, is_ask, ot, time_in_force, reduce_only, trigger_price, order_expiry, nonce, api_key_index),
        # Sometimes base_amount is float
        (market_index, client_order_index, float(base_amount), price, is_ask, ot, time_in_force, reduce_only, trigger_price, order_expiry, nonce, api_key_index),
    ]

    for args in candidates:
        try:
            out = await _maybe_await(fn(*args))
            # some SDKs return (signed, err)
            if isinstance(out, (list, tuple)) and len(out) == 2:
                signed, err = out
                if err is not None:
                    raise Exception(str(err))
                return signed
            return out
        except Exception as e:
            last_err = e

    sig = str(inspect.signature(fn))
    raise HTTPException(status_code=500, detail=f"Failed to call sign_create_order. signature={sig}. last_error={last_err}")


async def _send_tx(signer: Any, signed_tx: Any) -> Any:
    """
    Your SDK error shows: SignerClient.send_tx() got an unexpected keyword argument 'tx'
    So we send positionally first, then try other keyword names.
    """
    fn = getattr(signer, "send_tx", None) or getattr(signer, "sendTx", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose send_tx / sendTx")

    # 1) positional
    try:
        return await _maybe_await(fn(signed_tx))
    except Exception as e_pos:
        last_err = e_pos

    # 2) keyword variants
    for kw in ["transaction", "signed_tx", "raw_tx", "tx"]:
        try:
            return await _maybe_await(fn(**{kw: signed_tx}))
        except Exception as e_kw:
            last_err = e_kw

    sig = str(inspect.signature(fn))
    raise HTTPException(status_code=500, detail=f"Failed to send tx. signature={sig}. last_error={last_err}")


# -----------------------------
# main order endpoint
# -----------------------------
@app.post("/order")
async def place_order(req: OrderReq):
    signer = None
    api_client = None

    try:
        eth_private_key = _need("ETH_PRIVATE_KEY")
        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        market_index = _market_index_for(req.market)

        signer = make_signer_client()
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        # nonce (still from TransactionApi)
        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)
        nonce = await _call_next_nonce(tx_api, account_index, api_key_index)

        # BUY/SELL mapping
        side = 0 if req.side == "BUY" else 1
        client_order_index = int(time.time() * 1000)

        signed_tx = await _sign_create_order(
            signer=signer,
            eth_private_key=eth_private_key,
            market_index=market_index,
            side=side,
            base_amount=req.size,
            nonce=nonce,
            client_order_index=client_order_index,
            api_key_index=api_key_index,
            order_type=req.order_type,
        )

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed OK (not sent). Set live=true to send.",
                "market": req.market,
                "market_index": market_index,
                "side": req.side,
                "is_ask": True if side == 1 else False,
                "size": req.size,
                "base_amount_int": _int_if_close(req.size),
                "nonce": nonce,
                "client_order_index": client_order_index,
            }

        # LIVE send via signer (not tx_api)
        sent = await _send_tx(signer, signed_tx)

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
