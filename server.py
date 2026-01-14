import os
import time
import inspect
from typing import Any, Dict, Literal, Optional, Tuple

import lighter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


# -------------------- helpers --------------------
def _need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise HTTPException(status_code=500, detail=f"Missing {name}")
    return v


def _opt(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _as_int(name: str) -> int:
    try:
        return int(_need(name))
    except Exception:
        raise HTTPException(status_code=500, detail=f"{name} must be an integer")


def _as_float(name: str, default: float) -> float:
    v = _opt(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        raise HTTPException(status_code=500, detail=f"{name} must be a number")


def _strip_0x(s: str) -> str:
    return s[2:] if s.startswith("0x") else s


def _normalise_api_key_hex(s: str) -> str:
    """
    Lighter python lighter-sdk SignerClient expects API private key as:
      40 bytes = 80 hex chars
    (0x prefix is allowed in env; we remove it)
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


def _parse_market_index_map(raw: str) -> Dict[str, int]:
    """
    Accepts:
      - "BTC-USDC=1,ETH-USDC=2"
      - JSON-ish: {"BTC-USDC":1,"ETH-USDC":2}
    """
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        # tiny JSON parser without importing json (Render keeps it anyway, but this is fine)
        import json  # stdlib

        obj = json.loads(raw)
        return {str(k): int(v) for k, v in obj.items()}

    out: Dict[str, int] = {}
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _market_index_for(market: str) -> int:
    """
    Resolve market like "BTC-USDC" -> market_index.
    You must set env MARKET_INDEX_MAP like: BTC-USDC=1,ETH-USDC=2
    """
    raw = _opt("MARKET_INDEX_MAP", "")
    mp = _parse_market_index_map(raw or "")
    if market not in mp:
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve market_index for {market}. Set MARKET_INDEX_MAP env var like: BTC-USDC=1,ETH-USDC=2",
        )
    return int(mp[market])


def _maybe_await(x: Any):
    if inspect.isawaitable(x):
        return x
    return None


async def _await_if_needed(x: Any):
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


def _call_with_accepted_kwargs(fn, **kwargs):
    """
    Filter kwargs to only those accepted by the function signature.
    Works for both sync and async functions (we await later).
    """
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(**filtered)


# -------------------- request models --------------------
class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)

    # Lighter Python SDK behaviour: if you don't give a price, it effectively becomes 0 and gets rejected.
    # So we make price required for LIVE sends.
    price: Optional[float] = None

    # sign only vs send
    live: bool = False

    # Optional knobs (defaulted)
    order_type: int = 1  # 1 is common for LIMIT in many L2s; adjust if your SDK expects different
    time_in_force: int = 0  # 0 commonly "GTC"
    reduce_only: bool = False


# -------------------- endpoints --------------------
@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}


@app.get("/debug-env")
def debug_env():
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    api_key_raw = os.getenv("LIGHTER_API_KEY_PRIVATE_KEY", "")
    api_key_no0x = _strip_0x(api_key_raw).strip()
    eth_raw = os.getenv("ETH_PRIVATE_KEY", "")

    return {
        "BASE_URL": base_url,
        "LIGHTER_ACCOUNT_INDEX": os.getenv("LIGHTER_ACCOUNT_INDEX"),
        "LIGHTER_API_KEY_INDEX": os.getenv("LIGHTER_API_KEY_INDEX"),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_raw": len(api_key_raw),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_no0x": len(api_key_no0x),
        "ETH_PRIVATE_KEY_present": bool(eth_raw),
        "MARKET_INDEX_MAP_present": bool(_opt("MARKET_INDEX_MAP", "")),
        "BASE_AMOUNT_SCALE": _as_float("BASE_AMOUNT_SCALE", 1e8),
        "PRICE_SCALE": _as_float("PRICE_SCALE", 1e6),
    }


# -------------------- sdk adapters --------------------
async def _call_next_nonce(tx_api: Any, account_index: int, api_key_index: int) -> int:
    for name in ["next_nonce", "nextNonce"]:
        fn = getattr(tx_api, name, None)
        if fn:
            res = await _await_if_needed(_call_with_accepted_kwargs(fn, account_index=account_index, api_key_index=api_key_index))
            if hasattr(res, "nonce"):
                return int(res.nonce)
            if isinstance(res, dict) and "nonce" in res:
                return int(res["nonce"])
    raise HTTPException(status_code=500, detail="Could not find TransactionApi next_nonce method")


def _to_base_amount_int(size: float) -> int:
    scale = _as_float("BASE_AMOUNT_SCALE", 1e8)  # your earlier logs matched 0.001 -> 100000 with 1e8
    return int(round(size * scale))


def _to_price_int(price: float) -> int:
    scale = _as_float("PRICE_SCALE", 1e6)  # USDC commonly 1e6, but keep it configurable
    return int(round(price * scale))


async def _sign_create_order(
    signer: Any,
    market_index: int,
    api_key_index: int,
    base_amount_int: int,
    price_int: int,
    is_ask: bool,
    order_type: int,
    time_in_force: int,
    reduce_only: bool,
    nonce: int,
    client_order_index: int,
):
    """
    Your Render error printed a signature like:
      (market_index, client_order_index, base_amount, price, is_ask, order_type, time_in_force, reduce_only=False,
       trigger_price=0, order_expiry=-1, nonce:int=-1, api_key_index:int=255)

    So: no eth_private_key kw at all. The signer likely handles signing internally with the ETH key you provided elsewhere.
    We call it using only accepted kwargs to survive minor SDK changes.
    """
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose sign_create_order")

    # Provide more params than needed; we filter to only accepted ones
    kwargs = dict(
        market_index=market_index,
        client_order_index=client_order_index,
        base_amount=base_amount_int,
        price=price_int,
        is_ask=is_ask,
        order_type=order_type,
        time_in_force=time_in_force,
        reduce_only=reduce_only,
        nonce=nonce,
        api_key_index=api_key_index,
        trigger_price=0,
        order_expiry=-1,
    )

    try:
        out = await _await_if_needed(_call_with_accepted_kwargs(fn, **kwargs))
    except Exception as e:
        sig = str(inspect.signature(fn))
        raise HTTPException(status_code=500, detail=f"Failed to call sign_create_order. signature={sig}. error={e}")

    # Many versions return a 4-tuple: (tx_type, tx_info, signature, err)
    if isinstance(out, (list, tuple)) and len(out) >= 4:
        tx_type, tx_info, signature, err = out[0], out[1], out[2], out[3]
        if err is not None:
            raise HTTPException(status_code=500, detail=f"sign_create_order failed: {err}")
        return tx_type, tx_info, signature

    # Some versions may return an object/dict; pass it through and let send step handle
    return out


async def _send_signed_tx(signer: Any, tx_api: Any, signed_out: Any):
    """
    Your error showed:
      SignerClient.send_tx() got an unexpected keyword argument 'tx'
    so we must NOT call send_tx(tx=...).
    We attempt in this order:
      1) signer.send_tx(tx_type=?, tx_info=?, signature=?)
      2) signer.send_tx(tx_type, tx_info, signature) (positional)
      3) tx_api.send_tx(...) if it exists and accepts compatible args
    """
    # If sign returned the 3 pieces
    tx_type = tx_info = signature = None
    if isinstance(signed_out, (list, tuple)) and len(signed_out) == 3:
        tx_type, tx_info, signature = signed_out

    # 1) signer.send_tx
    send_fn = getattr(signer, "send_tx", None) or getattr(signer, "sendTx", None) or getattr(signer, "send_txn", None)
    if send_fn and tx_type is not None:
        # try kwargs (filtered)
        try:
            res = await _await_if_needed(_call_with_accepted_kwargs(send_fn, tx_type=tx_type, tx_info=tx_info, signature=signature))
            return res
        except Exception:
            # try positional
            try:
                res = await _await_if_needed(send_fn(tx_type, tx_info, signature))
                return res
            except Exception as e:
                sig = str(inspect.signature(send_fn))
                raise HTTPException(status_code=500, detail=f"Failed to send via signer.send_tx. signature={sig}. error={e}")

    # 2) TransactionApi send method (fallback)
    for name in ["send_tx", "sendTx"]:
        fn = getattr(tx_api, name, None)
        if fn:
            # if we have tx_type/tx_info/signature, try those; otherwise pass whole object
            try:
                if tx_type is not None:
                    return await _await_if_needed(_call_with_accepted_kwargs(fn, tx_type=tx_type, tx_info=tx_info, signature=signature))
                return await _await_if_needed(_call_with_accepted_kwargs(fn, tx=signed_out))
            except Exception as e:
                sig = str(inspect.signature(fn))
                raise HTTPException(status_code=500, detail=f"Failed to send via TransactionApi.{name}. signature={sig}. error={e}")

    raise HTTPException(status_code=500, detail="Could not find a working send_tx method on SignerClient or TransactionApi")


# -------------------- main order endpoint --------------------
@app.post("/order")
async def place_order(req: OrderReq):
    signer = None
    api_client = None

    try:
        # required for SDK initialisation / account ownership
        _ = _need("ETH_PRIVATE_KEY")

        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        # Resolve market_index via env map
        market_index = _market_index_for(req.market)

        # Convert amounts to int “lots”
        base_amount_int = _to_base_amount_int(req.size)

        # Price rules: required if live=true (otherwise Lighter rejects)
        if req.live and req.price is None:
            raise HTTPException(status_code=400, detail="price is required when live=true (Lighter rejects price=0)")

        price_int = _to_price_int(req.price or 0.0)

        # very basic guard to avoid the exact error you hit
        if req.live and price_int < 1:
            raise HTTPException(status_code=400, detail="price too low after scaling (must be >= 1 in integer units)")

        # BUY = not ask, SELL = ask
        is_ask = True if req.side == "SELL" else False

        signer = make_signer_client()

        # This verifies keys/nonce manager initialised
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)

        nonce = await _call_next_nonce(tx_api, account_index, api_key_index)
        client_order_index = int(time.time() * 1000)

        signed_out = await _sign_create_order(
            signer=signer,
            market_index=market_index,
            api_key_index=api_key_index,
            base_amount_int=base_amount_int,
            price_int=price_int,
            is_ask=is_ask,
            order_type=req.order_type,
            time_in_force=req.time_in_force,
            reduce_only=req.reduce_only,
            nonce=nonce,
            client_order_index=client_order_index,
        )

        # sign-only mode
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
                "price": req.price,
                "price_int": price_int,
                "nonce": nonce,
                "client_order_index": client_order_index,
            }

        # LIVE send
        sent = await _send_signed_tx(signer=signer, tx_api=tx_api, signed_out=signed_out)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "market_index": market_index,
            "side": req.side,
            "size": req.size,
            "base_amount_int": base_amount_int,
            "price": req.price,
            "price_int": price_int,
            "nonce": nonce,
            "client_order_index": client_order_index,
            "response": sent,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # close aiohttp sessions if present
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
