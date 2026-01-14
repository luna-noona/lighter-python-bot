import os
import time
import inspect
from typing import Any, Dict, Literal

import lighter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

# ---------------- helpers ----------------

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
    Lighter SDK expects API private key as:
    40 bytes = 80 hex characters (NO 0x)
    """
    h = _strip_0x(s).strip()
    try:
        int(h, 16)
    except Exception:
        raise HTTPException(status_code=500, detail="LIGHTER_API_KEY_PRIVATE_KEY is not valid hex")

    if len(h) != 80:
        raise HTTPException(
            status_code=500,
            detail=f"LIGHTER_API_KEY_PRIVATE_KEY must be 80 hex chars. Got {len(h)}"
        )
    return h


def make_signer_client() -> lighter.SignerClient:
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
    api_key_index = _as_int("LIGHTER_API_KEY_INDEX")
    api_private_key = _normalise_api_key_hex(_need("LIGHTER_API_KEY_PRIVATE_KEY"))

    return lighter.SignerClient(
        url=base_url,
        account_index=account_index,
        api_private_keys={api_key_index: api_private_key},
    )


def make_api_client() -> lighter.ApiClient:
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    return lighter.ApiClient(configuration=lighter.Configuration(host=base_url))


async def _maybe_await(x):
    if inspect.isawaitable(x):
        return await x
    return x

# ---------------- models ----------------

class OrderReq(BaseModel):
    market: str = Field(..., description="e.g. BTC-USDC")
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)
    price: float | None = None
    live: bool = False   # live=false = SAFE, live=true = REAL TRADE

# ---------------- endpoints ----------------

@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}


@app.get("/debug-env")
def debug_env():
    raw = os.getenv("LIGHTER_API_KEY_PRIVATE_KEY", "")
    return {
        "BASE_URL": os.getenv("BASE_URL"),
        "ACCOUNT_INDEX": os.getenv("LIGHTER_ACCOUNT_INDEX"),
        "API_KEY_INDEX": os.getenv("LIGHTER_API_KEY_INDEX"),
        "API_KEY_LEN": len(_strip_0x(raw)),
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
    }

# ---------------- SDK adapters ----------------

async def _next_nonce(tx_api, account_index, api_key_index) -> int:
    fn = getattr(tx_api, "next_nonce", None) or getattr(tx_api, "nextNonce", None)
    if not fn:
        raise HTTPException(status_code=500, detail="next_nonce not found")

    res = await _maybe_await(fn(account_index=account_index, api_key_index=api_key_index))
    return int(res.nonce if hasattr(res, "nonce") else res["nonce"])


async def _sign_create_order(
    signer,
    eth_private_key,
    market,
    side,
    base_amount,
    nonce,
    client_order_index,
    price=None,
):
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="sign_create_order not available")

    params = {
        "market": market,
        "side": side,
        "base_amount": base_amount,
        "nonce": nonce,
        "client_order_index": client_order_index,
    }

    if price is not None:
        params["price"] = price

    try:
        out = await _maybe_await(fn(**params))
    except TypeError:
        out = await _maybe_await(fn(eth_private_key, **params))

    if isinstance(out, (tuple, list)):
        signed, err = out
        if err:
            raise HTTPException(status_code=500, detail=str(err))
        return signed

    return out


async def _send_signed_tx(signer, signed_tx):
    """
    Correct send path for YOUR lighter-sdk:
    signer.send_tx(tx_type, tx_info)
    """
    if isinstance(signed_tx, (tuple, list)) and len(signed_tx) == 2:
        tx_type, tx_info = signed_tx
        return signer.send_tx(tx_type, tx_info)

    raise HTTPException(status_code=500, detail="Unexpected signed_tx format")

# ---------------- main order endpoint ----------------

@app.post("/order")
async def place_order(req: OrderReq):
    signer = None
    api_client = None

    try:
        eth_private_key = _need("ETH_PRIVATE_KEY")
        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        signer = make_signer_client()
        err = signer.check_client()
        if err:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)

        nonce = await _next_nonce(tx_api, account_index, api_key_index)
        side = 0 if req.side == "BUY" else 1
        client_order_index = int(time.time() * 1000)

        signed_tx = await _sign_create_order(
            signer=signer,
            eth_private_key=eth_private_key,
            market=req.market,
            side=side,
            base_amount=req.size,
            nonce=nonce,
            client_order_index=client_order_index,
            price=req.price,
        )

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed OK (not sent).",
                "market": req.market,
                "size": req.size,
                "nonce": nonce,
                "client_order_index": client_order_index,
            }

        # 🔥 REAL SEND 🔥
        sent = await _send_signed_tx(signer, signed_tx)

        return {
            "success": True,
            "live": True,
            "response": sent,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            if signer:
                await signer.close()
        except Exception:
            pass
        try:
            if api_client:
                await api_client.close()
        except Exception:
            pass
