import os
import time
from typing import Literal

import lighter
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# --------------------
# App
# --------------------
app = FastAPI()

# --------------------
# Config / helpers
# --------------------
def _need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise HTTPException(status_code=500, detail=f"Missing env var: {name}")
    return v


def _as_int(name: str) -> int:
    try:
        return int(_need(name))
    except ValueError:
        raise HTTPException(status_code=500, detail=f"{name} must be an integer")


def _auth(x_bot_token: str | None):
    expected = os.getenv("BOT_TOKEN")
    if expected and x_bot_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorised")


# --------------------
# Market mapping
# --------------------
# IMPORTANT: market_index is what Lighter actually uses
MARKET_INDEX_MAP = {
    "BTC-USDC": 1,
    "ETH-USDC": 2,
}

# --------------------
# Clients
# --------------------
def make_signer_client() -> lighter.SignerClient:
    return lighter.SignerClient(
        url=os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai"),
        account_index=_as_int("LIGHTER_ACCOUNT_INDEX"),
        api_private_keys={
            _as_int("LIGHTER_API_KEY_INDEX"): _need("LIGHTER_API_KEY_PRIVATE_KEY")
        },
    )


def make_api_client() -> lighter.ApiClient:
    return lighter.ApiClient(
        configuration=lighter.Configuration(
            host=os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
        )
    )


# --------------------
# Request model
# --------------------
class OrderReq(BaseModel):
    market: str = Field(..., example="BTC-USDC")
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)
    live: bool = False  # false = sign only, true = broadcast


# --------------------
# Routes
# --------------------
@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}


@app.post("/order")
async def place_order(
    req: OrderReq,
    x_bot_token: str | None = Header(default=None),
):
    _auth(x_bot_token)

    if req.market not in MARKET_INDEX_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported market. Use one of: {list(MARKET_INDEX_MAP.keys())}",
        )

    market_index = MARKET_INDEX_MAP[req.market]
    side_int = 0 if req.side == "BUY" else 1

    signer = make_signer_client()
    api = make_api_client()
    tx_api = lighter.TransactionApi(api)

    try:
        # Check signer is valid
        err = signer.check_client()
        if err is not None:
            raise HTTPException(status_code=500, detail=f"Signer error: {err}")

        # Get nonce
        nonce_resp = await tx_api.next_nonce(
            account_index=_as_int("LIGHTER_ACCOUNT_INDEX"),
            api_key_index=_as_int("LIGHTER_API_KEY_INDEX"),
        )
        nonce = int(nonce_resp.nonce)

        client_order_index = int(time.time() * 1000)

        # ---------
        # SIGN MARKET ORDER
        # ---------
        signed_tx = signer.sign_create_order(
            market_index,              # <-- POSITIONAL (this is the key fix)
            side_int,
            req.size,
            nonce,
            client_order_index,
        )

        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed OK (market order, not sent)",
                "market": req.market,
                "market_index": market_index,
                "side": req.side,
                "size": req.size,
                "nonce": nonce,
                "client_order_index": client_order_index,
            }

        # ---------
        # BROADCAST
        # ---------
        resp = await tx_api.send_tx(signed_tx)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "market_index": market_index,
            "side": req.side,
            "size": req.size,
            "nonce": nonce,
            "client_order_index": client_order_index,
            "response": resp,
        }

    finally:
        try:
            await signer.close()
        except Exception:
            pass
        try:
            await api.close()
        except Exception:
            pass
