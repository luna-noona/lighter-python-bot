import os
import time
from typing import Literal, Any

import lighter
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# ----------------------------
# App
# ----------------------------
app = FastAPI()

# ----------------------------
# Auth
# ----------------------------
def _auth(x_bot_token: str | None):
    expected = os.getenv("BOT_TOKEN")
    if expected and x_bot_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorised")

# ----------------------------
# Helpers
# ----------------------------
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
    h = _strip_0x(s).strip()
    int(h, 16)  # validate hex
    if len(h) != 80:
        raise HTTPException(
            status_code=500,
            detail="LIGHTER_API_KEY_PRIVATE_KEY must be 80 hex chars",
        )
    return h

# ----------------------------
# Lighter clients
# ----------------------------
def make_signer() -> lighter.SignerClient:
    return lighter.SignerClient(
        url=os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai"),
        account_index=_as_int("LIGHTER_ACCOUNT_INDEX"),
        api_private_keys={
            _as_int("LIGHTER_API_KEY_INDEX"): _normalise_api_key_hex(
                _need("LIGHTER_API_KEY_PRIVATE_KEY")
            )
        },
    )

def make_api() -> lighter.ApiClient:
    return lighter.ApiClient(
        configuration=lighter.Configuration(
            host=os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
        )
    )

# ----------------------------
# Models
# ----------------------------
class OrderReq(BaseModel):
    market: str
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)
    live: bool = False  # false = sign only, true = broadcast

# ----------------------------
# Health
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}

# ----------------------------
# Order (MARKET ONLY)
# ----------------------------
@app.post("/order")
async def place_order(
    req: OrderReq,
    x_bot_token: str | None = Header(default=None),
):
    _auth(x_bot_token)

    signer = None
    api_client = None

    try:
        signer = make_signer()
        err = signer.check_client()
        if err:
            raise HTTPException(status_code=500, detail=str(err))

        api_client = make_api()
        tx_api = lighter.TransactionApi(api_client)

        # nonce
        nonce_resp = await tx_api.next_nonce(
            account_index=_as_int("LIGHTER_ACCOUNT_INDEX"),
            api_key_index=_as_int("LIGHTER_API_KEY_INDEX"),
        )
        nonce = int(nonce_resp.nonce)

        side_int = 0 if req.side == "BUY" else 1
        client_order_index = int(time.time() * 1000)

        # 🔑 MARKET ORDER — NO PRICE, NO ETH KEY PASSED
        signed_tx = signer.sign_create_order(
            market=req.market,
            side=side_int,
            base_amount=req.size,
            nonce=nonce,
            client_order_index=client_order_index,
        )

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed OK (market order, not sent)",
                "market": req.market,
                "side": req.side,
                "size": req.size,
                "nonce": nonce,
                "client_order_index": client_order_index,
            }

        # Broadcast
        send_resp = await tx_api.send_tx(signed_tx)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "side": req.side,
            "size": req.size,
            "nonce": nonce,
            "client_order_index": client_order_index,
            "response": send_resp,
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
