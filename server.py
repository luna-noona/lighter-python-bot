import os
import time
import asyncio
from typing import Optional, Literal, Any, Dict

import lighter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


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
    Lighter python SignerClient expects API private key as *40 bytes* (80 hex chars).
    Some people paste with 0x prefix; remove it.
    Keep it as 80 hex chars (DO NOT trim to 64).
    """
    h = _strip_0x(s).strip()
    # sanity: hex only
    try:
        int(h, 16)
    except Exception:
        raise HTTPException(status_code=500, detail="LIGHTER_API_KEY_PRIVATE_KEY is not valid hex")
    if len(h) != 80:
        raise HTTPException(status_code=500, detail=f"LIGHTER_API_KEY_PRIVATE_KEY must be 80 hex chars (40 bytes). Got {len(h)}")
    return h


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


class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)

    # SAFETY SWITCH:
    # live=false => just builds/signs and returns debug (no send)
    # live=true  => actually sends tx
    live: bool = False


@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}


@app.get("/debug-env")
def debug_env():
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    api_key_raw = os.getenv("LIGHTER_API_KEY_PRIVATE_KEY", "")
    api_key_no0x = _strip_0x(api_key_raw).strip()

    out: Dict[str, Any] = {
        "BASE_URL": base_url,
        "LIGHTER_ACCOUNT_INDEX": os.getenv("LIGHTER_ACCOUNT_INDEX"),
        "LIGHTER_API_KEY_INDEX": os.getenv("LIGHTER_API_KEY_INDEX"),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_raw": len(api_key_raw),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_no0x": len(api_key_no0x),
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
    }
    return out


async def _call_next_nonce(tx_api: Any, account_index: int, api_key_index: int) -> int:
    """
    Different builds name this slightly differently. Try a few.
    """
    for name in ["next_nonce", "nextNonce"]:
        fn = getattr(tx_api, name, None)
        if fn:
            res = await fn(account_index=account_index, api_key_index=api_key_index)
            # res might be an object with .nonce or dict-like
            if hasattr(res, "nonce"):
                return int(res.nonce)
            if isinstance(res, dict) and "nonce" in res:
                return int(res["nonce"])
            # fallthrough if unexpected
    raise HTTPException(status_code=500, detail="Could not find a working TransactionApi next_nonce method")


async def _call_sign_create_order(
    signer: Any,
    eth_private_key: str,
    market: str,
    side: int,
    base_amount: float,
    nonce: int,
    client_order_index: int,
) -> Any:
    """
    Try common method names used by the SDK.
    """
    for name in ["sign_create_order", "signCreateOrder", "sign_create_order_tx", "signCreateOrderTx"]:
        fn = getattr(signer, name, None)
        if fn:
            # Some versions return (signed, err), some just signed
            out = await fn(
                eth_private_key=eth_private_key,
                market=market,
                side=side,
                base_amount=base_amount,
                nonce=nonce,
                client_order_index=client_order_index,
            )
            return out
    raise HTTPException(status_code=500, detail="Could not find a working SignerClient sign_create_order method")


async def _call_send_tx(tx_api: Any, tx: Any) -> Any:
    for name in ["send_tx", "sendTx"]:
        fn = getattr(tx_api, name, None)
        if fn:
            return await fn(tx=tx)
    raise HTTPException(status_code=500, detail="Could not find a working TransactionApi send_tx method")


@app.post("/order")
async def place_order(req: OrderReq):
    """
    Market order test:
    - live=false => just initialises + gets nonce + signs (no send)
    - live=true  => sends to Lighter
    """
    signer = None
    api_client = None

    try:
        eth_private_key = _need("ETH_PRIVATE_KEY")
        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        # 1) init signer client
        signer = make_signer_client()
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        # 2) nonce
        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)
        nonce = await _call_next_nonce(tx_api, account_index, api_key_index)

        # 3) sign order
        side = 0 if req.side == "BUY" else 1
        client_order_index = int(time.time() * 1000)

        signed_out = await _call_sign_create_order(
            signer=signer,
            eth_private_key=eth_private_key,
            market=req.market,
            side=side,
            base_amount=req.size,
            nonce=nonce,
            client_order_index=client_order_index,
        )

        # normalise outputs across SDK versions
        signed_tx = signed_out
        sign_err = None
        if isinstance(signed_out, (list, tuple)) and len(signed_out) == 2:
            signed_tx, sign_err = signed_out
        if sign_err is not None:
            raise HTTPException(status_code=500, detail=f"sign_create_order failed: {sign_err}")

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed OK (not sent). Set live=true to send.",
                "nonce": nonce,
                "client_order_index": client_order_index,
                "market": req.market,
                "side": req.side,
                "size": req.size,
            }

        # 4) send tx (LIVE)
        sent = await _call_send_tx(tx_api, signed_tx)

        return {
            "success": True,
            "live": True,
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
