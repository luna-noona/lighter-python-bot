import os
import time
import inspect
from typing import Any, Dict, Literal, Optional, Tuple

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
    Lighter python SignerClient expects API private key as 40 bytes = 80 hex chars.
    Allow optional 0x prefix; remove it.
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


# -----------------------------
# clients
# -----------------------------
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


# -----------------------------
# request models
# -----------------------------
class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)

    # live=false => sign only
    # live=true  => send (real)
    live: bool = False

    # For your SDK this seems required (you discovered order_type=1 fixes the int/float error)
    order_type: int = 1


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
        "BASE_AMOUNT_MULT": os.getenv("BASE_AMOUNT_MULT", "100000000"),
    }


# -----------------------------
# market mapping
# -----------------------------
def _market_index_for(market: str) -> int:
    """
    Provide MARKET_INDEX_MAP env var like:
      BTC-USDC=1,ETH-USDC=2
    """
    raw = os.getenv("MARKET_INDEX_MAP", "").strip()
    if not raw:
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve market_index for {market}. Set MARKET_INDEX_MAP env var like: BTC-USDC=1,ETH-USDC=2",
        )

    mapping: Dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            continue
        mapping[k] = int(v)

    if market not in mapping:
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve market_index for {market}. MARKET_INDEX_MAP missing {market}. Example: BTC-USDC=1,ETH-USDC=2",
        )
    return mapping[market]


# -----------------------------
# amount conversion
# -----------------------------
def _base_amount_int(size: float) -> int:
    """
    Your SignerClient signature wants base_amount as int.
    Default multiplier 1e8; override via BASE_AMOUNT_MULT if needed.
    """
    mult = int(os.getenv("BASE_AMOUNT_MULT", "100000000"))  # 1e8 default
    return int(round(size * mult))


# -----------------------------
# sdk adapters
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


def _extract_tx_type_info(signed_out: Any) -> Tuple[int, str]:
    """
    Your SDK’s send_tx expects (tx_type: int, tx_info: str).

    We accept:
    - (tx_type, tx_info)
    - {"tx_type": ..., "tx_info": ...}
    - object with .tx_type / .tx_info
    """
    if isinstance(signed_out, (list, tuple)) and len(signed_out) == 2:
        tx_type, tx_info = signed_out
        return int(tx_type), str(tx_info)

    if isinstance(signed_out, dict) and "tx_type" in signed_out and "tx_info" in signed_out:
        return int(signed_out["tx_type"]), str(signed_out["tx_info"])

    if hasattr(signed_out, "tx_type") and hasattr(signed_out, "tx_info"):
        return int(getattr(signed_out, "tx_type")), str(getattr(signed_out, "tx_info"))

    raise HTTPException(
        status_code=500,
        detail=f"Could not extract (tx_type, tx_info) from sign_create_order output. Got type={type(signed_out)}",
    )


async def _sign_create_order(
    signer: Any,
    market_index: int,
    client_order_index: int,
    base_amount_int: int,
    is_ask: bool,
    order_type: int,
    nonce: int,
    api_key_index: int,
) -> Tuple[int, str]:
    """
    From your observed signature:
      sign_create_order(
        market_index, client_order_index, base_amount, price, is_ask,
        order_type, time_in_force, reduce_only, trigger_price,
        order_expiry, nonce, api_key_index
      )
    We'll do a market order style: price=0, trigger_price=0, order_expiry=-1, tif=0, reduce_only=0
    """
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose sign_create_order")

    price = 0
    time_in_force = 0
    reduce_only = 0
    trigger_price = 0
    order_expiry = -1

    args = (
        int(market_index),
        int(client_order_index),
        int(base_amount_int),
        int(price),
        bool(is_ask),
        int(order_type),
        int(time_in_force),
        int(reduce_only),
        int(trigger_price),
        int(order_expiry),
        int(nonce),
        int(api_key_index),
    )

    out = await _maybe_await(fn(*args))

    # Some versions return (result, err)
    if isinstance(out, (list, tuple)) and len(out) == 2 and not isinstance(out[0], (int, str)):
        maybe_res, maybe_err = out
        if maybe_err is not None:
            raise HTTPException(status_code=500, detail=f"sign_create_order failed: {maybe_err}")
        out = maybe_res

    return _extract_tx_type_info(out)


async def _send_tx(signer: Any, tx_type: int, tx_info: str) -> Any:
    """
    Your SDK signature looks like send_tx(tx_type: int, tx_info: str).
    """
    fn = getattr(signer, "send_tx", None) or getattr(signer, "sendTx", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose send_tx / sendTx")

    return await _maybe_await(fn(int(tx_type), str(tx_info)))


# -----------------------------
# main order endpoint
# -----------------------------
@app.post("/order")
async def place_order(req: OrderReq):
    signer = None
    api_client = None

    try:
        _need("ETH_PRIVATE_KEY")  # presence check (your SignerClient uses env key material internally)
        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        market_index = _market_index_for(req.market)
        is_ask = True if req.side == "SELL" else False

        signer = make_signer_client()
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)
        nonce = await _call_next_nonce(tx_api, account_index, api_key_index)

        client_order_index = int(time.time() * 1000)
        base_amount_int = _base_amount_int(req.size)

        tx_type, tx_info = await _sign_create_order(
            signer=signer,
            market_index=market_index,
            client_order_index=client_order_index,
            base_amount_int=base_amount_int,
            is_ask=is_ask,
            order_type=int(req.order_type),
            nonce=nonce,
            api_key_index=api_key_index,
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
                "base_amount_int": base_amount_int,
                "nonce": nonce,
                "client_order_index": client_order_index,
                "tx_type": tx_type,
                "tx_info_len": len(tx_info),
            }

        # LIVE send
        sent = await _send_tx(signer, tx_type, tx_info)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "market_index": market_index,
            "nonce": nonce,
            "client_order_index": client_order_index,
            "tx_type": tx_type,
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
