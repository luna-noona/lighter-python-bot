import os
import time
import inspect
from typing import Any, Dict, Literal, Optional, Tuple, Union

import lighter
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

# Allow Loveable/browser calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers
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
            detail=f"LIGHTER_API_KEY_PRIVATE_KEY must be 80 hex chars (40 bytes). Got {len(h)}",
        )
    return h


def _parse_map(env_name: str) -> Dict[str, str]:
    """
    Parse maps like: "BTC-USDC=1,ETH-USDC=2"
    Returns dict[str,str] with trimmed keys/values.
    """
    raw = os.getenv(env_name, "").strip()
    out: Dict[str, str] = {}
    if not raw:
        return out
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _auth(x_bot_token: Optional[str]) -> None:
    expected = os.getenv("BOT_TOKEN")
    # If BOT_TOKEN is not set, don't block (dev mode)
    if expected and x_bot_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorised")


async def _maybe_await(x: Any) -> Any:
    return await x if inspect.isawaitable(x) else x


def make_signer_client() -> lighter.SignerClient:
    base_url = os.getenv("BASE_URL", os.getenv("BASE_API_URL", "https://mainnet.zklighter.elliot.ai"))
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
    base_url = os.getenv("BASE_URL", os.getenv("BASE_API_URL", "https://mainnet.zklighter.elliot.ai"))
    return lighter.ApiClient(configuration=lighter.Configuration(host=base_url))


# -----------------------------
# Models
# -----------------------------
class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0, description="Base asset size, e.g. 0.0001 BTC")
    price: float = Field(..., gt=0, description="Required by Lighter for market-like orders")
    live: bool = False


# -----------------------------
# Tx extraction + broadcasting
# -----------------------------
def _normalise_signed_output(raw: Any) -> Dict[str, Any]:
    """
    Try to extract tx_type + tx_info from whatever the SDK returns.
    We keep the raw response for debugging too.
    """
    tx_type = None
    tx_info = None

    # dict shapes
    if isinstance(raw, dict):
        if "tx_type" in raw and "tx_info" in raw:
            tx_type = raw.get("tx_type")
            tx_info = raw.get("tx_info")
        elif "signed" in raw and isinstance(raw["signed"], dict):
            s = raw["signed"]
            if "tx_type" in s and "tx_info" in s:
                tx_type = s.get("tx_type")
                tx_info = s.get("tx_info")

    # tuple/list shapes
    if (tx_type is None or tx_info is None) and isinstance(raw, (list, tuple)):
        # common: (signed, err)
        if len(raw) == 2 and raw[1] is None:
            return _normalise_signed_output(raw[0])

        # sometimes: (tx_type, tx_info, ...)
        if len(raw) >= 2:
            a, b = raw[0], raw[1]
            # tx_type might be int/str, tx_info often dict/str
            if isinstance(a, (int, str)) and isinstance(b, (dict, str)):
                tx_type, tx_info = a, b

        # or tx packaged inside any element dict
        if tx_type is None or tx_info is None:
            for item in raw:
                if isinstance(item, dict) and "tx_type" in item and "tx_info" in item:
                    tx_type = item.get("tx_type")
                    tx_info = item.get("tx_info")
                    break

    return {
        "tx_type": tx_type,
        "tx_info": tx_info,
        "raw": raw,
    }


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


async def _broadcast(tx_api: Any, tx_type: Any, tx_info: Any) -> Any:
    """
    Newer lighter builds want: send_tx(tx_type=?, tx_info=?)
    Older builds sometimes want: send_tx(tx=?)
    We try a few.
    """
    send_fn = getattr(tx_api, "send_tx", None) or getattr(tx_api, "sendTx", None)
    if not send_fn:
        raise HTTPException(status_code=500, detail="TransactionApi has no send_tx method")

    last_err = None

    # Attempt 1: keyword tx_type/tx_info
    try:
        return await _maybe_await(send_fn(tx_type=tx_type, tx_info=tx_info))
    except Exception as e:
        last_err = e

    # Attempt 2: positional (tx_type, tx_info)
    try:
        return await _maybe_await(send_fn(tx_type, tx_info))
    except Exception as e:
        last_err = e

    # Attempt 3: single kw "tx" packed
    try:
        return await _maybe_await(send_fn(tx={"tx_type": tx_type, "tx_info": tx_info}))
    except Exception as e:
        last_err = e

    # Attempt 4: single positional packed
    try:
        return await _maybe_await(send_fn({"tx_type": tx_type, "tx_info": tx_info}))
    except Exception as e:
        last_err = e

    raise HTTPException(status_code=500, detail=f"Could not broadcast tx. last_error={last_err}")


def _to_base_amount(market: str, size: float) -> int:
    """
    Convert size float to base_amount int using AMOUNT_SCALE_MAP if present,
    otherwise BASE_DECIMALS_MAP (scale = 10**decimals).
    """
    scale_map = _parse_map("AMOUNT_SCALE_MAP")  # e.g. BTC-USDC=100000000
    if market in scale_map:
        scale = int(scale_map[market])
        return int(round(size * scale))

    dec_map = _parse_map("BASE_DECIMALS_MAP")  # e.g. BTC-USDC=8
    if market in dec_map:
        decimals = int(dec_map[market])
        return int(round(size * (10 ** decimals)))

    # fallback (safe-ish)
    return int(round(size * (10 ** 8)))


def _to_price_int(price: float) -> int:
    """
    Your logs show Lighter expecting an int and erroring on float.
    For now we treat price as an integer number of quote units (e.g. 45000).
    """
    return int(price)


def _market_index_for(market: str) -> int:
    mi = _parse_map("MARKET_INDEX_MAP")  # BTC-USDC=1
    if market not in mi:
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve market_index for {market}. Set MARKET_INDEX_MAP env var like: BTC-USDC=1,ETH-USDC=2",
        )
    return int(mi[market])


def _sign_create_order_positional(
    signer: Any,
    market_index: int,
    client_order_index: int,
    base_amount: int,
    price_int: int,
    is_ask: bool,
    nonce: int,
    api_key_index: int,
) -> Any:
    """
    Calls the signature your logs show (positional):
      sign_create_order(market_index, client_order_index, base_amount, price, is_ask,
                        order_type, time_in_force, reduce_only=False, trigger_price=0,
                        order_expiry=-1, nonce=-1, api_key_index=255)
    We set:
      order_type=1  (market-like)
      time_in_force=0
      reduce_only=False
      trigger_price=0
      order_expiry=-1
    """
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose sign_create_order")

    order_type = 1          # market-like (based on your earlier success with order_type=1)
    time_in_force = 0
    reduce_only = False
    trigger_price = 0
    order_expiry = -1

    # Use positional to avoid kw mismatch across builds
    return fn(
        market_index,
        client_order_index,
        base_amount,
        price_int,
        is_ask,
        order_type,
        time_in_force,
        reduce_only,
        trigger_price,
        order_expiry,
        nonce,
        api_key_index,
    )


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}


@app.get("/debug-env")
def debug_env():
    # Do NOT leak secrets; just show presence/lengths
    api_key_raw = os.getenv("LIGHTER_API_KEY_PRIVATE_KEY", "")
    api_key_no0x = _strip_0x(api_key_raw).strip()
    return {
        "BASE_URL": os.getenv("BASE_URL", os.getenv("BASE_API_URL", "https://mainnet.zklighter.elliot.ai")),
        "LIGHTER_ACCOUNT_INDEX": os.getenv("LIGHTER_ACCOUNT_INDEX"),
        "LIGHTER_API_KEY_INDEX": os.getenv("LIGHTER_API_KEY_INDEX"),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_raw": len(api_key_raw),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_no0x": len(api_key_no0x),
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
        "BOT_TOKEN_set": bool(os.getenv("BOT_TOKEN")),
        "MARKET_INDEX_MAP_preview": os.getenv("MARKET_INDEX_MAP", "")[:80],
        "BASE_DECIMALS_MAP_preview": os.getenv("BASE_DECIMALS_MAP", "")[:80],
        "AMOUNT_SCALE_MAP_preview": os.getenv("AMOUNT_SCALE_MAP", "")[:80],
    }


@app.post("/order")
async def place_order(
    req: OrderReq,
    x_bot_token: Optional[str] = Header(default=None, convert_underscores=False),
):
    _auth(x_bot_token)

    signer = None
    api_client = None

    try:
        # env requirements
        _need("ETH_PRIVATE_KEY")  # signer uses this internally depending on build
        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        market_index = _market_index_for(req.market)
        base_amount = _to_base_amount(req.market, req.size)
        price_int = _to_price_int(req.price)

        # BUY => is_ask False, SELL => is_ask True
        is_ask = True if req.side == "SELL" else False

        signer = make_signer_client()
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)

        nonce = await _call_next_nonce(tx_api, account_index, api_key_index)
        client_order_index = int(time.time() * 1000)

        raw_signed = _sign_create_order_positional(
            signer=signer,
            market_index=market_index,
            client_order_index=client_order_index,
            base_amount=base_amount,
            price_int=price_int,
            is_ask=is_ask,
            nonce=nonce,
            api_key_index=api_key_index,
        )

        signed_norm = _normalise_signed_output(raw_signed)
        tx_type = signed_norm.get("tx_type")
        tx_info = signed_norm.get("tx_info")

        # If not live: just return what we'd broadcast
        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed market-like order OK (not sent). Set live=true to broadcast.",
                "market": req.market,
                "market_index": market_index,
                "side": req.side,
                "is_ask": is_ask,
                "size": req.size,
                "base_amount": base_amount,
                "price": req.price,
                "order_price_int": price_int,
                "nonce": nonce,
                "client_order_index": client_order_index,
                "tx_type": tx_type,
                "tx_info_present": tx_info is not None,
                "signed_raw": raw_signed,
            }

        # Live: require broadcast-ready payload
        if tx_type is None or tx_info is None:
            raise HTTPException(
                status_code=500,
                detail="Signed output did not include tx_type/tx_info, cannot broadcast with this lighter-sdk build.",
            )

        sent = await _broadcast(tx_api, tx_type, tx_info)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "market_index": market_index,
            "side": req.side,
            "is_ask": is_ask,
            "size": req.size,
            "base_amount": base_amount,
            "price": req.price,
            "order_price_int": price_int,
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
