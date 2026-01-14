import os
import time
from typing import Literal

import lighter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()


# ---------------- helpers ----------------
def need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise HTTPException(status_code=500, detail=f"Missing {name}")
    return v


def as_int(name: str) -> int:
    try:
        return int(need(name))
    except Exception:
        raise HTTPException(status_code=500, detail=f"{name} must be an integer")


def strip_0x(s: str) -> str:
    return s[2:] if s.startswith("0x") else s


def normalise_api_key_hex(s: str) -> str:
    h = strip_0x(s).strip()
    try:
        int(h, 16)
    except Exception:
        raise HTTPException(status_code=500, detail="API key is not valid hex")
    if len(h) != 80:
        raise HTTPException(
            status_code=500,
            detail=f"LIGHTER_API_KEY_PRIVATE_KEY must be 80 hex chars, got {len(h)}",
        )
    return h


def market_index_for(market: str) -> int:
    raw = os.getenv("MARKET_INDEX_MAP", "")
    mapping = dict(
        item.split("=") for item in raw.split(",") if "=" in item
    )
    if market not in mapping:
        raise HTTPException(
            status_code=500,
            detail=f"Unknown market {market}. Set MARKET_INDEX_MAP (e.g. BTC-USDC=1)",
        )
    return int(mapping[market])


# ---------------- clients ----------------
def make_signer() -> lighter.SignerClient:
    return lighter.SignerClient(
        url=os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai"),
        account_index=as_int("LIGHTER_ACCOUNT_INDEX"),
        api_private_keys={
            as_int("LIGHTER_API_KEY_INDEX"):
            normalise_api_key_hex(need("LIGHTER_API_KEY_PRIVATE_KEY"))
        },
    )


def make_api():
    return lighter.ApiClient(
        configuration=lighter.Configuration(
            host=os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
        )
    )


# ---------------- request ----------------
class OrderReq(BaseModel):
    market: str
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)
    price: float | None = None
    live: bool = False


# ---------------- endpoints ----------------
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}


@app.post("/order")
async def order(req: OrderReq):
    signer = None
    api = None

    try:
        # required
        need("ETH_PRIVATE_KEY")

        market_index = market_index_for(req.market)
        base_amount = int(req.size * 1e8)   # confirmed correct from your logs
        price_int = int((req.price or 0) * 1e6)

        if req.live and price_int < 1:
            raise HTTPException(400, "price too low")

        signer = make_signer()
        err = signer.check_client()
        if err:
            raise HTTPException(500, f"check_client failed: {err}")

        api = make_api()
        tx_api = lighter.TransactionApi(api)

        nonce_res = await tx_api.next_nonce(
            account_index=as_int("LIGHTER_ACCOUNT_INDEX"),
            api_key_index=as_int("LIGHTER_API_KEY_INDEX"),
        )
        nonce = int(nonce_res.nonce)

        client_order_index = int(time.time() * 1000)
        is_ask = req.side == "SELL"

        # 🔑 THIS CALL IS NOW CORRECT
        tx_type, tx_info, sig, err = signer.sign_create_order(
            market_index=market_index,
            client_order_index=client_order_index,
            base_amount=base_amount,
            price=price_int,
            is_ask=is_ask,
            order_type=1,
            time_in_force=0,
            reduce_only=False,
            trigger_price=0,
            order_expiry=0,     # ✅ FIX
            nonce=nonce,
            api_key_index=as_int("LIGHTER_API_KEY_INDEX"),
        )

        if err:
            raise HTTPException(500, f"sign_create_order failed: {err}")

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed OK (not sent)",
                "market": req.market,
                "nonce": nonce,
                "client_order_index": client_order_index,
            }

        # send
        res = signer.send_tx(tx_type, tx_info, sig)

        return {
            "success": True,
            "live": True,
            "response": res,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if signer:
            await signer.close()
        if api:
            await api.close()
