import os
import time
from typing import Optional

import lighter
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------------------------------
# App setup
# -------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Environment
# -------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
LIGHTER_API_KEY_INDEX = int(os.getenv("LIGHTER_API_KEY_INDEX", "0"))

# -------------------------------------------------
# Models
# -------------------------------------------------

class OrderReq(BaseModel):
    market: str = Field(..., example="BTC-USDC")
    side: str = Field(..., example="BUY")  # BUY or SELL
    size: float = Field(..., gt=0)
    live: bool = False


# -------------------------------------------------
# Auth
# -------------------------------------------------

def _auth(x_bot_token: Optional[str]):
    if BOT_TOKEN and x_bot_token != BOT_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorised")


# -------------------------------------------------
# Helpers
# -------------------------------------------------

MARKET_INDEX_MAP = {
    "BTC-USDC": 1,
    "ETH-USDC": 2,
}

def get_market_index(market: str) -> int:
    if market not in MARKET_INDEX_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown market {market}"
        )
    return MARKET_INDEX_MAP[market]


# -------------------------------------------------
# Health
# -------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}


# -------------------------------------------------
# Order endpoint (MARKET ONLY)
# -------------------------------------------------

@app.post("/order")
async def place_order(
    req: OrderReq,
    x_bot_token: Optional[str] = Header(default=None),
):
    _auth(x_bot_token)

    market_index = get_market_index(req.market)

    signer = None
    client = None

    try:
        signer = lighter.SignerClient.from_env()
        client = lighter.ApiClient.from_env(signer)

        # MARKET ORDER ONLY
        tx, tx_hash, err = await client.create_order(
            market_index=market_index,
            side=req.side,
            size=req.size,
            order_type="MARKET",
            time_in_force="FOK",  # Fill-or-kill
            api_key_index=LIGHTER_API_KEY_INDEX,
        )

        if err:
            raise HTTPException(status_code=500, detail=str(err))

        response = {
            "success": True,
            "live": req.live,
            "market": req.market,
            "market_index": market_index,
            "side": req.side,
            "size": req.size,
            "nonce": tx.nonce if hasattr(tx, "nonce") else None,
            "client_order_index": tx.client_order_index if hasattr(tx, "client_order_index") else None,
        }

        # Broadcast only if live
        if req.live:
            send_result = await client.send_tx(tx)
            response["response"] = send_result
        else:
            response["message"] = "Signed OK (not sent). Set live=true to broadcast."

        return response

    finally:
        if signer:
            await signer.close()
        if client:
            await client.close()
