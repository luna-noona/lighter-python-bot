import os
import time
import inspect
from decimal import Decimal
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
    Parses env like: "BTC-USDC=1,ETH-USDC=2"
    """
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise HTTPException(status_code=500, detail=f"{env_name} has invalid entry: {part}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _market_index(market: str) -> int:
    m = _parse_map("MARKET_INDEX_MAP")
    if market not in m:
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve market_index for {market}. Set MARKET_INDEX_MAP like: BTC-USDC=1,ETH-USDC=2",
        )
    return int(m[market])


def _amount_scale(market: str) -> int:
    m = _parse_map("AMOUNT_SCALE_MAP")
    if market not in m:
        raise HTTPException(
            status_code=500,
            detail=f"Missing amount scale for {market}. Set AMOUNT_SCALE_MAP like: BTC-USDC=100000000",
        )
    return int(m[market])


def _to_base_amount_int(market: str, size: float) -> int:
    """
    Convert human size (float) -> integer base_amount expected by lighter-sdk.
    Uses AMOUNT_SCALE_MAP per market (e.g. BTC-USDC=100000000).
    """
    scale = _amount_scale(market)
    # use Decimal to avoid float rounding weirdness
    amt = (Decimal(str(size)) * Decimal(scale)).to_integral_value()
    return int(amt)


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
        "MARKET_INDEX_MAP": bool(os.getenv("MARKET_INDEX_MAP")),
        "AMOUNT_SCALE_MAP": bool(os.getenv("AMOUNT_SCALE_MAP")),
    }


@app.post("/order")
async def place_order(req: OrderReq):
    signer = None

    try:
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        signer = make_signer_client()
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"check_client failed: {err}")

        market_index = _market_index(req.market)

        # BUY => is_ask=False, SELL => is_ask=True
        is_ask = True if req.side == "SELL" else False

        base_amount = _to_base_amount_int(req.market, req.size)

        # For market orders in this SDK, set:
        # order_type = 1 (market) as you’ve been using
        order_type = 1

        # Common default time_in_force for market orders:
        # 0 usually = IOC in many SDKs; if Lighter differs, the error will tell us.
        time_in_force = 0

        client_order_index = int(time.time() * 1000)

        # price/is_ask/order_type/time_in_force are required by your SDK signature
        # price for market order -> 0
        signed = await _maybe_await(
            signer.sign_create_order(
                market_index=market_index,
                client_order_index=client_order_index,
                base_amount=base_amount,
                price=0,
                is_ask=is_ask,
                order_type=order_type,
                time_in_force=time_in_force,
                api_key_index=api_key_index,
            )
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
                "base_amount_int": base_amount,
                "client_order_index": client_order_index,
            }

        # LIVE send: your SDK often includes a method on signer for sending;
        # if not, we’ll add TransactionApi in the next step.
        send_fn = getattr(signer, "send_tx", None) or getattr(signer, "sendTx", None)
        if not send_fn:
            raise HTTPException(
                status_code=500,
                detail="This lighter-sdk build does not expose send_tx on SignerClient. Tell me your /order sign-only response and I’ll wire TransactionApi.send_tx correctly.",
            )

        sent = await _maybe_await(send_fn(tx=signed))

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "market_index": market_index,
            "side": req.side,
            "base_amount_int": base_amount,
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
