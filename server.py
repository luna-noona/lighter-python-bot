import os
import time
import inspect
from typing import Any, Dict, Literal, Optional

import lighter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


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
    Lighter SignerClient (python lighter-sdk) expects API private key as:
      40 bytes = 80 hex chars
    (Some UIs show 32 bytes/64 hex; YOUR python SDK build expects 80.)
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


def _parse_market_index_map() -> Dict[str, int]:
    """
    MARKET_INDEX_MAP format:
      BTC-USDC=1,ETH-USDC=2
    """
    raw = os.getenv("MARKET_INDEX_MAP", "").strip()
    mapping: Dict[str, int] = {}
    if not raw:
        return mapping

    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        mapping[k.strip()] = int(v.strip())
    return mapping


def _resolve_market_index(market: str) -> Optional[int]:
    return _parse_market_index_map().get(market)


# -----------------------------
# Clients
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
# Request model
# -----------------------------
class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)

    # Your current lighter-sdk build signs a limit-style order and requires a price.
    price: Optional[float] = None

    # live=false => signs only (safe, no trade)
    # live=true  => attempts broadcast (live trading)
    live: bool = False


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
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
        "MARKET_INDEX_MAP_present": bool(os.getenv("MARKET_INDEX_MAP")),
        "MARKET_INDEX_MAP_preview": os.getenv("MARKET_INDEX_MAP", "")[:120],
    }


@app.get("/sdk-info")
def sdk_info():
    """
    Shows what your deployed lighter-sdk build exposes (so we can wire broadcast precisely).
    """
    import inspect as _inspect

    sc = lighter.SignerClient
    tx = lighter.TransactionApi

    def _sig(cls, name: str):
        fn = getattr(cls, name, None)
        if not fn:
            return None
        try:
            return str(_inspect.signature(fn))
        except Exception:
            return "signature unavailable"

    signer_methods = sorted({m for m in dir(sc) if any(k in m.lower() for k in ["sign", "order", "tx", "send", "broadcast"])})
    tx_methods = sorted({m for m in dir(tx) if any(k in m.lower() for k in ["nonce", "tx", "send", "broadcast"])})

    return {
        "lighter_version": getattr(lighter, "__version__", "unknown"),
        "signerclient_methods_filtered": signer_methods,
        "transactionapi_methods_filtered": tx_methods,
        "signatures": {
            "SignerClient.sign_create_order": _sig(sc, "sign_create_order"),
            "SignerClient.send_tx": _sig(sc, "send_tx"),
            "TransactionApi.next_nonce": _sig(tx, "next_nonce"),
            "TransactionApi.send_tx": _sig(tx, "send_tx"),
            "TransactionApi.sendTx": _sig(tx, "sendTx"),
        },
    }


# -----------------------------
# SDK adapters
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


async def _sign_create_order(signer: Any, req: OrderReq, nonce: int, client_order_index: int) -> Any:
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose sign_create_order")

    market_index = _resolve_market_index(req.market)
    if market_index is None:
        raise HTTPException(
            status_code=500,
            detail=f'Unknown market "{req.market}". Set MARKET_INDEX_MAP like "BTC-USDC=1,ETH-USDC=2".',
        )

    if req.price is None:
        raise HTTPException(status_code=500, detail="price is required for this SDK build")

    # Lighter is_ask convention: SELL=True, BUY=False
    is_ask = (req.side == "SELL")

    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())

    # This matches the signature you’re seeing in your logs.
    candidate: Dict[str, Any] = {
        "market_index": int(market_index),
        "client_order_index": int(client_order_index),
        "index": 0,
        "base_amount": float(req.size),
        "price": int(req.price),
        "is_ask": bool(is_ask),
        "order_type": 1,       # limit
        "time_in_force": 0,    # GTC
        "reduce_only": False,
        "trigger_price": 0,
        "order_expiry": 0,
        "nonce": int(nonce),
        "api_key_index": _as_int("LIGHTER_API_KEY_INDEX"),
    }

    filtered = {k: v for k, v in candidate.items() if k in accepted}

    try:
        return await _maybe_await(fn(**filtered))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"sign_create_order failed. signature={sig}. accepted={sorted(list(accepted))}. error={e}",
        )


def _extract_tx_payload(signed_out: Any) -> Dict[str, Any]:
    """
    Try to extract {tx_type, tx_info} for broadcast. SDK variants differ.
    """
    if isinstance(signed_out, (list, tuple)) and len(signed_out) >= 2:
        return {"tx_type": signed_out[0], "tx_info": signed_out[1]}

    if isinstance(signed_out, dict):
        if "tx_type" in signed_out and "tx_info" in signed_out:
            return {"tx_type": signed_out["tx_type"], "tx_info": signed_out["tx_info"]}
        return signed_out

    for a in ["tx_type", "txType"]:
        if hasattr(signed_out, a):
            tx_type = getattr(signed_out, a)
            tx_info = getattr(signed_out, "tx_info", None) or getattr(signed_out, "txInfo", None)
            return {"tx_type": tx_type, "tx_info": tx_info}

    raise HTTPException(
        status_code=500,
        detail="Could not extract (tx_type, tx_info) from sign_create_order output. Run /sdk-info and paste it.",
    )


async def _broadcast_best_effort(signer: Any, tx_api: Any, signed_out: Any) -> Any:
    payload = _extract_tx_payload(signed_out)

    # 1) TransactionApi send
    for name in ["send_tx", "sendTx"]:
        fn = getattr(tx_api, name, None)
        if not fn:
            continue

        # try kwargs
        try:
            return await _maybe_await(fn(**payload))
        except TypeError:
            pass

        # try positional: (tx_type, tx_info)
        if "tx_type" in payload and "tx_info" in payload:
            try:
                return await _maybe_await(fn(payload["tx_type"], payload["tx_info"]))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"{name} positional failed: {e}")

    # 2) SignerClient send
    for name in ["send_tx", "sendTx", "send"]:
        fn = getattr(signer, name, None)
        if not fn:
            continue

        try:
            return await _maybe_await(fn(**payload))
        except TypeError:
            pass

        if "tx_type" in payload and "tx_info" in payload:
            try:
                return await _maybe_await(fn(payload["tx_type"], payload["tx_info"]))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"{name} positional failed: {e}")

    raise HTTPException(
        status_code=500,
        detail="No working broadcast method found in this lighter-sdk build. Run /sdk-info and paste output.",
    )


# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/order")
async def place_order(req: OrderReq):
    signer = None
    api_client = None

    try:
        # keep present; needed for real trading later even if not passed into sign_create_order
        _need("ETH_PRIVATE_KEY")

        signer = make_signer_client()
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)

        nonce = await _call_next_nonce(tx_api, _as_int("LIGHTER_ACCOUNT_INDEX"), _as_int("LIGHTER_API_KEY_INDEX"))
        client_order_index = int(time.time() * 1000)

        signed_out = await _sign_create_order(signer, req, nonce, client_order_index)

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed OK (not sent). Set live=true to attempt broadcast.",
                "market": req.market,
                "market_index": _resolve_market_index(req.market),
                "side": req.side,
                "size": req.size,
                "price": req.price,
                "nonce": nonce,
                "client_order_index": client_order_index,
            }

        sent = await _broadcast_best_effort(signer, tx_api, signed_out)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "market_index": _resolve_market_index(req.market),
            "side": req.side,
            "size": req.size,
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
