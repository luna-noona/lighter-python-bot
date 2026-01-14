import os
import time
import inspect
from typing import Any, Dict, Literal

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
    Lighter SignerClient (python lighter-sdk) expects API private key as:
      40 bytes = 80 hex chars
    Render env var can include optional 0x prefix; we remove it.
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


async def _maybe_await(x):
    """Await if x is awaitable, else return x."""
    if inspect.isawaitable(x):
        return await x
    return x


# ---------- request models ----------
class OrderReq(BaseModel):
    market: str = Field(..., description='e.g. "BTC-USDC"')
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)
    live: bool = False  # live=false => sign only; live=true => send tx


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
        "LIGHTER_API_KEY_PRIVATE_KEY_len_raw": len(api_key_raw),
        "LIGHTER_API_KEY_PRIVATE_KEY_len_no0x": len(api_key_no0x),
        "ETH_PRIVATE_KEY_present": bool(os.getenv("ETH_PRIVATE_KEY")),
    }


# ---------- sdk adapters (handle version differences) ----------
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


async def _sign_create_order(
    signer: Any,
    eth_private_key: str,
    market: str,
    side: int,
    base_amount: float,
    nonce: int,
    client_order_index: int,
):
    """
    Handles multiple lighter-sdk variants for sign_create_order:
      - sync or async
      - may accept ETH key as kwarg, positional, or not at all
      - market/size/base_amount naming differences
    """
    fn = getattr(signer, "sign_create_order", None) or getattr(signer, "signCreateOrder", None)
    if not fn:
        raise HTTPException(status_code=500, detail="SignerClient does not expose sign_create_order")

    sig = inspect.signature(fn)

    field_variants = [
        {"market": market, "side": side, "base_amount": base_amount, "nonce": nonce, "client_order_index": client_order_index},
        {"ticker": market, "side": side, "base_amount": base_amount, "nonce": nonce, "client_order_index": client_order_index},
        {"market": market, "side": side, "size": base_amount, "nonce": nonce, "client_order_index": client_order_index},
        {"ticker": market, "side": side, "size": base_amount, "nonce": nonce, "client_order_index": client_order_index},
    ]

    # Try order: no key -> positional key -> kw key
    key_kw_variants = [
        None,  # no key
        {"eth_private_key": eth_private_key},
        {"private_key": eth_private_key},
        {"key": eth_private_key},
        {"signer_private_key": eth_private_key},
    ]

    last_err = None

    for fields in field_variants:
        # 1) Try without key kw
        try:
            out = await _maybe_await(fn(**fields))
            # normalise (signed, err) tuples
            if isinstance(out, (list, tuple)) and len(out) == 2:
                signed, err = out
                if err is not None:
                    raise Exception(str(err))
                return signed
            return out
        except Exception as e:
            last_err = e

        # 2) Try ETH key as positional first arg (some builds do this)
        try:
            out = await _maybe_await(fn(eth_private_key, **fields))
            if isinstance(out, (list, tuple)) and len(out) == 2:
                signed, err = out
                if err is not None:
                    raise Exception(str(err))
                return signed
            return out
        except Exception as e:
            last_err = e

        # 3) Try various kwarg names for the key
        for key_kw in key_kw_variants:
            if not key_kw:
                continue
            try:
                out = await _maybe_await(fn(**{**fields, **key_kw}))
                if isinstance(out, (list, tuple)) and len(out) == 2:
                    signed, err = out
                    if err is not None:
                        raise Exception(str(err))
                    return signed
                return out
            except Exception as e:
                last_err = e

    raise HTTPException(
        status_code=500,
        detail=f"Failed to call sign_create_order. signature={sig}. last_error={last_err}",
    )


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
        # required for signing/sending
        eth_private_key = _need("ETH_PRIVATE_KEY")
        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        signer = make_signer_client()
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)

        nonce = await _call_next_nonce(tx_api, account_index, api_key_index)

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
        )

        # sign-only mode
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

        # LIVE send
        sent = await _send_tx(tx_api, signed_tx)

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
