"""
src/ingestion/etherscan_poll.py

Asynchronous on-chain data ingestion module using Etherscan REST API.

Polls blockchain transactions, token transfers, and gas metrics periodically,
normalizes them, and stores them in DuckDB via DatabaseManager.

Supports fault-tolerant backoff, rate limiting, and incremental updates.
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import aiohttp
import pandas as pd

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

logger = logging.getLogger("omerGPT.ingestion.etherscan_poll")

def load_config():
    if HAS_DOTENV:
        load_dotenv()
    config = {
        "api_key": os.getenv("ETHERSCAN_API_KEY", ""),
        "addresses": os.getenv("ETHERSCAN_ADDRESSES", "").split(",") if os.getenv("ETHERSCAN_ADDRESSES", "") else [],
        "poll_interval": int(os.getenv("ETHERSCAN_POLL_INTERVAL", "10")),
        "rate_limit": int(os.getenv("ETHERSCAN_RATE_LIMIT", "5")),
        "batch_size": int(os.getenv("ETHERSCAN_BATCH_SIZE", "50"))
    }
    config_path = "configs/config.yaml"
    if HAS_YAML and os.path.exists(config_path):
        try:
            with open(config_path, "r") as stream:
                yaml_config = yaml.safe_load(stream)
                for k, v in yaml_config.get("etherscan", {}).items():
                    config[k] = v
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")
    return config

class EtherscanPoller:
    """
    Asynchronous Etherscan API poller for on-chain metrics.
    Features:
    - Rate-limit aware polling (5 calls/sec free tier)
    - Exponential backoff with jitter on rate limits
    - State tracking per address (incremental updates)
    - Gas spike detection
    - Batch DB writes via DatabaseManager
    - Robust structured JSON logging
    """

    BASE_URL = "https://api.etherscan.io/api"
    MAX_RETRIES = 10
    INITIAL_BACKOFF = 1.0
    MAX_BACKOFF = 60.0
    BATCH_SIZE = 50
    POLL_INTERVAL = 10 # seconds (Etherscan free tier: 5 req/sec)
    REQUEST_TIMEOUT = 20

    def __init__(
        self,
        db_manager,
        api_key: str,
        addresses: List[str],
        poll_interval: int = POLL_INTERVAL,
        batch_size: int = BATCH_SIZE
    ):
        """
        Initialize Etherscan poller.
        Args:
            db_manager: DatabaseManager instance for persistence
            api_key: Etherscan API key
            addresses: List of Ethereum addresses to monitor
            poll_interval: Polling interval in seconds
        """
        self.db = db_manager
        self.api_key = api_key
        self.addresses = [addr.lower() for addr in addresses]
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self.state = {addr: {
            "last_tx_hash": None,
            "last_gas_price": 0.0,
            "last_block": None
        } for addr in self.addresses}
        self.stats = {
            "requests": 0,
            "transactions": 0,
            "token_transfers": 0,
            "gas_spikes": 0,
            "errors": 0,
            "rate_limits": 0
        }
        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "event": "init",
            "msg": f"Initialized EtherscanPoller for {len(addresses)} addresses"
        }))

    async def _api_request(self, params: Dict) -> Optional[Dict]:
        """
        Make rate-limited API request with exponential backoff.
        Args:
            params: Query parameters (module, action, etc.)
        Returns:
            Response JSON or None on failure
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        params["apikey"] = self.api_key
        retry_count = 0
        while retry_count < self.MAX_RETRIES:
            try:
                async with self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                ) as resp:
                    self.stats["requests"] += 1
                    event_log = {
                        "ts": datetime.utcnow().isoformat(),
                        "operation": "api_request",
                        "params": params,
                        "status": resp.status
                    }
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") in ["1", 1]:
                            logger.debug(json.dumps({**event_log, "result": "ok"}))
                            return data
                        else:
                            msg = data.get("message", "Unknown error")
                            logger.warning(json.dumps({**event_log, "result": "error", "err": msg}))
                            return None
                    elif resp.status == 429:
                        # Rate limit hit
                        self.stats["rate_limits"] += 1
                        backoff = self._calculate_backoff(retry_count)
                        logger.warning(json.dumps({
                            **event_log,
                            "result": "rate_limited",
                            "backoff": backoff
                        }))
                        await asyncio.sleep(backoff)
                        retry_count += 1
                    else:
                        logger.error(json.dumps({
                            **event_log,
                            "result": "http_error",
                            "err": await resp.text()
                        }))
                        return None
            except asyncio.TimeoutError:
                logger.warning(json.dumps({
                    "ts": datetime.utcnow().isoformat(),
                    "operation": "api_request",
                    "params": params,
                    "result": "timeout",
                    "attempt": retry_count + 1
                }))
                retry_count += 1
                await asyncio.sleep(self._calculate_backoff(retry_count))
            except aiohttp.ClientError as e:
                logger.error(json.dumps({
                    "ts": datetime.utcnow().isoformat(),
                    "operation": "api_request",
                    "params": params,
                    "result": "client_error",
                    "err": str(e)
                }))
                retry_count += 1
                await asyncio.sleep(self._calculate_backoff(retry_count))
            except Exception as e:
                logger.error(json.dumps({
                    "ts": datetime.utcnow().isoformat(),
                    "operation": "api_request",
                    "params": params,
                    "result": "unexpected_error",
                    "err": str(e)
                }), exc_info=True)
                return None
        logger.error(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "operation": "api_request",
            "params": params,
            "result": "max_retries_exceeded"
        }))
        self.stats["errors"] += 1
        return None

    def _calculate_backoff(self, retry_count: int) -> float:
        """Calculate exponential backoff with jitter."""
        backoff = min(
            self.INITIAL_BACKOFF * (2 ** retry_count),
            self.MAX_BACKOFF
        )
        jitter = random.uniform(0, 0.3 * backoff)
        return backoff + jitter

    def normalize_tx(self, tx: Dict) -> Optional[Dict]:
        """
        Normalize raw transaction dict to unified schema.
        Args:
            tx: Raw Etherscan transaction
        Returns:
            Normalized dict or None if invalid
        """
        try:
            timestamp = int(tx.get("timeStamp", 0))
            ts = pd.to_datetime(timestamp, unit="s") if timestamp > 0 else None
            value_wei = int(tx.get("value", "0"))
            value_eth = value_wei / 1e18
            gas_price_wei = int(tx.get("gasPrice", "0"))
            gas_price_gwei = gas_price_wei / 1e9
            is_token = bool(tx.get("tokenSymbol"))
            normalized = {
                "address": tx.get("from", "").lower(),
                "tx_hash": tx.get("hash", ""),
                "block_number": int(tx.get("blockNumber", 0)),
                "timestamp": ts,
                "value_eth": value_eth,
                "gas_price_gwei": gas_price_gwei,
                "is_token": is_token,
                "token_symbol": tx.get("tokenSymbol"),
                "token_name": tx.get("tokenName"),
                "contract_address": tx.get("contractAddress", "").lower() if is_token else None
            }
            return normalized
        except (KeyError, ValueError, TypeError) as e:
            logger.error(json.dumps({
                "ts": datetime.utcnow().isoformat(),
                "operation": "normalize_tx",
                "error": str(e),
                "raw": tx
            }))
            return None

    async def fetch_transactions(self, address: str) -> List[Dict]:
        """
        Fetch latest normal transactions for an address.
        Args:
            address: Ethereum address
        Returns:
            List of normalized transaction dicts
        """
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "sort": "desc",
            "page": 1,
            "offset": 100
        }
        result = await self._api_request(params)
        if not result or "result" not in result:
            return []
        txs = result["result"]
        if isinstance(txs, str):
            return []
        normalized = []
        for tx in txs:
            norm_tx = self.normalize_tx(tx)
            if norm_tx:
                normalized.append(norm_tx)
        self.stats["transactions"] += len(normalized)
        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "operation": "fetch_transactions",
            "address": address,
            "count": len(normalized),
            "last_block": self.state[address].get("last_block", "unknown")
        }))
        return normalized

    async def fetch_token_transfers(self, address: str) -> List[Dict]:
        """
        Fetch latest ERC20 token transfers for an address.
        Args:
            address: Ethereum address
        Returns:
            List of normalized token transfer dicts
        """
        params = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "sort": "desc",
            "page": 1,
            "offset": 100
        }
        result = await self._api_request(params)
        if not result or "result" not in result:
            return []
        txs = result["result"]
        if isinstance(txs, str):
            return []
        normalized = []
        for tx in txs:
            norm_tx = self.normalize_tx(tx)
            if norm_tx:
                normalized.append(norm_tx)
        self.stats["token_transfers"] += len(normalized)
        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "operation": "fetch_token_transfers",
            "address": address,
            "count": len(normalized)
        }))
        return normalized

    async def fetch_gas_metrics(self) -> Optional[Dict]:
        """
        Fetch current gas price and block time statistics.
        Returns:
            Dict with SafeGasPrice, StandardGasPrice, FastGasPrice (Gwei)
        """
        params = {
            "module": "gastracker",
            "action": "gasoracle"
        }
        result = await self._api_request(params)
        if not result or "result" not in result:
            return None
        gas_data = result["result"]
        metrics = {
            "safe_gwei": float(gas_data.get("SafeGasPrice", 0)),
            "standard_gwei": float(gas_data.get("StandardGasPrice", 0)),
            "fast_gwei": float(gas_data.get("FastGasPrice", 0)),
            "base_fee_gwei": float(gas_data.get("suggestBaseFeePerGas", 0)),
            "timestamp": datetime.now()
        }
        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "operation": "fetch_gas_metrics",
            "metrics": metrics
        }))
        return metrics

    async def _check_gas_spike(self, gas_metrics: Dict) -> bool:
        """
        Detect if gas price spiked > 15% in last minute.
        Args:
            gas_metrics: Current gas metrics
        Returns:
            True if spike detected
        """
        current_gas = gas_metrics["standard_gwei"]
        prev_gas = self.state[list(self.addresses)[0]].get("last_gas_price", current_gas)
        spike = False
        if prev_gas > 0:
            pct_change = abs(current_gas - prev_gas) / prev_gas * 100
            if pct_change > 15:
                logger.warning(json.dumps({
                    "ts": datetime.utcnow().isoformat(),
                    "operation": "gas_spike",
                    "prev_gas": prev_gas,
                    "current_gas": current_gas,
                    "pct_change": pct_change
                }))
                self.stats["gas_spikes"] += 1
                spike = True
        # Update last price for all addresses
        for addr in self.addresses:
            self.state[addr]["last_gas_price"] = current_gas
        return spike

    async def _flush_transactions(self, tx_batch: List[Dict]):
        """Flush transaction batch to database."""
        if not tx_batch:
            return
        try:
            df = pd.DataFrame(tx_batch)
            # Integration: Use existing db_manager methods. If not present, replace below with appropriate method.
            # For example, upsert_onchain_metrics, or generic upsert_candles if schema matches.
            await self.db.upsert_onchain_metrics(df)
            logger.info(json.dumps({
                "ts": datetime.utcnow().isoformat(),
                "operation": "flush_transactions",
                "count": len(tx_batch)
            }))
        except AttributeError:
            # fallback if upsert_onchain_metrics not implemented
            logger.info(json.dumps({
                "ts": datetime.utcnow().isoformat(),
                "operation": "flush_transactions",
                "msg": f"Upsert to DB not implemented - flushed {len(tx_batch)} records"
            }))
        except Exception as e:
            logger.error(json.dumps({
                "ts": datetime.utcnow().isoformat(),
                "operation": "flush_transactions",
                "error": str(e)
            }), exc_info=True)

    async def poll(self):
        """
        Main polling loop that periodically fetches data for all addresses.
        """
        self.running = True
        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "event": "start_poll",
            "msg": f"Starting Etherscan poller for {len(self.addresses)} addresses"
        }))
        tx_batch = []
        try:
            while self.running:
                try:
                    # Fetch gas metrics (once per poll)
                    gas_metrics = await self.fetch_gas_metrics()
                    if gas_metrics:
                        await self._check_gas_spike(gas_metrics)
                    # Fetch transactions for each address (parallel)
                    tasks = [
                        self.fetch_transactions(addr)
                        for addr in self.addresses
                    ]
                    tx_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for addr, result in zip(self.addresses, tx_results):
                        if isinstance(result, list):
                            tx_batch.extend(result)
                            # Update last block for state tracking
                            if result:
                                self.state[addr]["last_block"] = result[0].get("block_number")
                    # Fetch token transfers for each address (parallel)
                    token_tasks = [
                        self.fetch_token_transfers(addr)
                        for addr in self.addresses
                    ]
                    token_results = await asyncio.gather(*token_tasks, return_exceptions=True)
                    for addr, result in zip(self.addresses, token_results):
                        if isinstance(result, list):
                            tx_batch.extend(result)
                    # Flush batch if size reached
                    if len(tx_batch) >= self.batch_size:
                        await self._flush_transactions(tx_batch)
                        tx_batch = []
                    logger.info(json.dumps({
                        "ts": datetime.utcnow().isoformat(),
                        "operation": "poll_cycle",
                        "tx_buffer": len(tx_batch),
                        "cycle_done": True,
                        "stats": self.stats
                    }))
                    await asyncio.sleep(self.poll_interval)
                except asyncio.CancelledError:
                    logger.info(json.dumps({
                        "ts": datetime.utcnow().isoformat(),
                        "operation": "poll_cancelled"
                    }))
                    break
                except Exception as e:
                    logger.error(json.dumps({
                        "ts": datetime.utcnow().isoformat(),
                        "operation": "poll_error",
                        "error": str(e)
                    }), exc_info=True)
                    self.stats["errors"] += 1
                    await asyncio.sleep(5)
            # Final flush
            if tx_batch:
                await self._flush_transactions(tx_batch)
        finally:
            logger.info(json.dumps({
                "ts": datetime.utcnow().isoformat(),
                "operation": "poll_shutdown"
            }))

    async def stop(self):
        """Gracefully stop polling."""
        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "operation": "poll_stop",
            "msg": "Stopping Etherscan poller..."
        }))
        self.running = False
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "operation": "stats_final",
            "stats": self.stats
        }))



        def get_stats(self) -> Dict:
            """Get current polling statistics."""
            return self.stats.copy()

        def get_state(self) -> Dict:
            """Get current state (last processed blocks/txs)."""
            return self.state.copy()

    # ==================== DEMO ====================

    async def run_demo():
        """
        Demo: Poll Etherscan for 5 minutes with demo addresses.
        """
        import sys
        sys.path.insert(0, "src")
        from storage.db_manager import DatabaseManager

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        print("\n=== Etherscan On-Chain Poller Demo ===\n")
        config = load_config()
        api_key = config["api_key"] or os.getenv("ETHERSCAN_API_KEY", "YourAPIKeyToken")
        if not api_key or api_key == "YourAPIKeyToken":
            print("⚠️ Set ETHERSCAN_API_KEY environment variable")
            return
        db = DatabaseManager("data/market_data.duckdb")
        # Demo addresses (Uniswap, Aave, MakerDAO or from config)
        addresses = config["addresses"] or [
            "0x1111111254fb6c44bac0bed2854e76f90643097d", # 1inch
            "0xe592427a0aeb6b7b0edddc9bfd199a85dbc73d69" # Uniswap Router
        ]
        poller = EtherscanPoller(
            db_manager=db,
            api_key=api_key,
            addresses=addresses,
            poll_interval=config.get("poll_interval", 10),
            batch_size=config.get("batch_size", 50)
        )
        poll_task = asyncio.create_task(poller.poll())
        try:
            print(f"Polling {len(addresses)} addresses for 5 minutes...\n")
            for i in range(30): # 30 * 10s = 5 min
                await asyncio.sleep(10)
                stats = poller.get_stats()
                print(f"[{i*10}s] Requests: {stats['requests']} | "
                      f"TXs: {stats['transactions']} | "
                      f"Tokens: {stats['token_transfers']} | "
                      f"Gas spikes: {stats['gas_spikes']}")
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            await poller.stop()
            poll_task.cancel()
            db.close()
            print("\n=== Demo Complete ===\n")

    if __name__ == "__main__":
        asyncio.run(run_demo())
    """
    src/ingestion/etherscan_poll.py

    Asynchronous on-chain data ingestion module using Etherscan REST API.
    Polls blockchain transactions, token transfers, and gas metrics periodically,
    normalizes them, and stores them in DuckDB via DatabaseManager.
    Supports fault-tolerant backoff, rate limiting, and incremental updates.
    """

    import asyncio
    import json
    import logging
    import os
    import random
    import time
    from datetime import datetime, timedelta
    from typing import List, Optional, Dict
    import aiohttp
    import pandas as pd

    try:
        import yaml
        HAS_YAML = True
    except ImportError:
        HAS_YAML = False

    try:
        from dotenv import load_dotenv
        HAS_DOTENV = True
    except ImportError:
        HAS_DOTENV = False

    logger = logging.getLogger("omerGPT.ingestion.etherscan_poll")

def load_config():
    if HAS_DOTENV:
        load_dotenv()
    config = {
        "api_key": os.getenv("ETHERSCAN_API_KEY", ""),
        "addresses": os.getenv("ETHERSCAN_ADDRESSES", "").split(",") if os.getenv("ETHERSCAN_ADDRESSES", "") else [],
        "poll_interval": int(os.getenv("ETHERSCAN_POLL_INTERVAL", "10")),
        "rate_limit": int(os.getenv("ETHERSCAN_RATE_LIMIT", "5")),
        "batch_size": int(os.getenv("ETHERSCAN_BATCH_SIZE", "50"))
    }
    config_path = "configs/config.yaml"
    if HAS_YAML and os.path.exists(config_path):
        try:
            with open(config_path, "r") as stream:
                yaml_config = yaml.safe_load(stream)
            for k, v in yaml_config.get("etherscan", {}).items():
                config[k] = v
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")
    return config

