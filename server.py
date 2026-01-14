import os
import time
from typing import Literal, Optional

import lighter
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

# -----------------------------
# CORS (Loveable needs this)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Auth
# -----------------------------
def _auth(x_bot_token: Optional[str]):
    expected = os.getenv("BOT_TOKEN")
    if expected and x_bot_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorised")

# -----------------------------
# Helpers
# -----------------------------
def _need(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise HTTPException(status_code=500, detail=f"Missing env var: {name}")
    return v

def _as_int(name: str) -> int:
    return int(_need(name))

def _strip_0x(s: str) -> str:
    return s[2:] if s.startswith("0x") else s

def _normalise_api_key_hex(s: str) -> str:
    h = _strip_0x(s).strip()
    int(h, 16)
    if len(h) != 80:
        raise HTTPException(
            status_code=500,
            detail="LIGHTER_API_KEY_PRIVATE_KEY must be 80 hex chars",
        )
    return h

# -----------------------------
# Clients
# -----------------------------
def make_signer_client() -> lighter.SignerClient:
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")

    api_key_index = _as_int("LIGHTER_API_KEY_INDEX")
    account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
    api_private_key = _normalise_api_key_hex(
        _need("LIGHTER_API_KEY_PRIVATE_KEY")
    )

    return lighter.SignerClient(
        url=base_url,
        account_index=account_index,
        api_private_keys={api_key_index: api_private_key},
    )

def make_api_client() -> lighter.ApiClient:
    base_url = os.getenv("BASE_URL", "https://mainnet.zklighter.elliot.ai")
    return lighter.ApiClient(
        configuration=lighter.Configuration(host=base_url)
    )

# -----------------------------
# Models
# -----------------------------
class OrderReq(BaseModel):
    market: str = Field(..., example="BTC-USDC")
    side: Literal["BUY", "SELL"]
    size: float = Field(..., gt=0)
    live: bool = False  # false = paper, true = real

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "timestamp": int(time.time())}

# -----------------------------
# Order (MARKET ONLY)
# -----------------------------
@app.post("/order")
async def place_order(
    req: OrderReq,
    x_bot_token: Optional[str] = Header(default=None),
):
    _auth(x_bot_token)

    signer = None
    api_client = None

    try:
        eth_private_key = _need("ETH_PRIVATE_KEY")
        account_index = _as_int("LIGHTER_ACCOUNT_INDEX")
        api_key_index = _as_int("LIGHTER_API_KEY_INDEX")

        signer = make_signer_client()
        err = signer.check_client()
        if err:
            raise HTTPException(status_code=500, detail=str(err))

        api_client = make_api_client()
        tx_api = lighter.TransactionApi(api_client)

        # --- resolve market index ---
        market_map = os.getenv("MARKET_INDEX_MAP", "BTC-USDC=1")
        mapping = dict(x.split("=") for x in market_map.split(","))
        if req.market not in mapping:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown market {req.market}",
            )
        market_index = int(mapping[req.market])

        # --- nonce ---
        nonce_resp = await tx_api.next_nonce(
            account_index=account_index,
            api_key_index=api_key_index,
        )
        nonce = int(nonce_resp.nonce)

        # --- market order params ---
        is_ask = req.side == "SELL"
        base_amount = int(req.size * 1_000_000_000)  # base units
        client_order_index = int(time.time() * 1000)

        signed_tx = signer.sign_create_order(
            market_index=market_index,
            client_order_index=client_order_index,
            base_amount=base_amount,
            price=0,                 # MARKET ORDER
            is_ask=is_ask,
            order_type=0,            # MARKET
            time_in_force=0,
            reduce_only=False,
            trigger_price=0,
            order_expiry=0,
            nonce=nonce,
            api_key_index=api_key_index,
            eth_private_key=eth_private_key,
        )

        # paper mode
        if not req.live:
            return {
                "success": True,
                "live": False,
                "message": "Signed OK (not sent)",
                "market": req.market,
                "side": req.side,
                "size": req.size,
                "nonce": nonce,
                "client_order_index": client_order_index,
            }

        # broadcast
        resp = await tx_api.send_tx(signed_tx)

        return {
            "success": True,
            "live": True,
            "market": req.market,
            "side": req.side,
            "size": req.size,
            "nonce": nonce,
            "client_order_index": client_order_index,
            "response": resp,
        }

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
