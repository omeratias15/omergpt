"""
src/core/agent.py

OmerGPT Autonomous Agent.

Integrates LLM reasoning with live market & on-chain data.
Provides insights, strategy evaluation, and adaptive behavior
based on real-time signals and historical context.

Acts as conversational interface and analytical brain.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("omerGPT.agent")


class OmerGPTAgent:
    """
    Autonomous trading agent with LLM reasoning.
    
    Features:
    - Real-time signal analysis using LLM
    - Historical trade reflection and learning
    - Natural language interaction
    - Adaptive strategy evaluation
    - Conversational market insights
    """
    
    def __init__(
        self,
        db_manager,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        reflection_interval: int = 600
    ):
        """
        Initialize OmerGPT agent.
        
        Args:
            db_manager: DatabaseManager instance
            model_name: LLM model (gpt-4, gpt-4-turbo, etc.)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            reflection_interval: Seconds between autonomous reflections
        """
        self.db = db_manager
        self.model_name = model_name
        self.reflection_interval = reflection_interval
        
        # Initialize OpenAI client
        try:
            from openai import AsyncOpenAI
            
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️ No OpenAI API key provided. Agent will run in read-only mode.")
                self.client = None
            else:
                self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            logger.warning("⚠️ OpenAI not installed. Install with: pip install openai")
            self.client = None
        
        # State tracking
        self.running = False
        self.conversation_history: List[Dict] = []
        self.insights_log: List[Dict] = []
        
        logger.info(
            f"Agent initialized: model={model_name}, "
            f"reflection_interval={reflection_interval}s"
        )

    async def analyze_signals(self, limit: int = 10) -> str:
        """
        Analyze recent trading signals using LLM.
        
        Args:
            limit: Number of recent signals to analyze
            
        Returns:
            Analysis text
        """
        if not self.client:
            return "LLM client not available. Running analysis with basic heuristics..."
        
        try:
            # Fetch recent signals
            query = """
                SELECT symbol, ts_ms, signal_type, confidence, reason, anomaly_score
                FROM signals
                WHERE status = 'new'
                ORDER BY ts_ms DESC
                LIMIT ?
            """
            
            result = self.db.conn.execute(query, (limit,))
            signals_df = result.df()
            
            if signals_df.empty:
                return "No recent signals to analyze."
            
            # Prepare context
            signals_text = self._format_dataframe_for_llm(signals_df)
            
            # Get current market snapshot
            market_context = await self._get_market_context()
            
            # Create prompt
            system_prompt = """You are OmerGPT, an advanced AI trading assistant with deep knowledge of:
- Technical analysis (RSI, volatility, momentum indicators)
- On-chain metrics (gas prices, whale transfers, inflows/outflows)
- Anomaly detection and pattern recognition
- Risk management and portfolio theory

Analyze the provided trading signals and market context. Provide:
1. Summary of key signals
2. Pattern recognition (if any)
3. Risk assessment
4. Actionable insights
5. Recommended monitoring areas

Be concise but comprehensive. Use specific numbers and data points."""
            
            user_message = f"""Analyze these recent trading signals:

{signals_text}

Market Context:
{market_context}

Provide a focused analysis with actionable insights."""
            
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            analysis = response.choices[0].message.content
            
            # Log insight
            self._log_insight("signal_analysis", analysis, signals_df)
            
            logger.info(f"✓ Signal analysis complete ({len(signals_df)} signals)")
            return analysis
        
        except Exception as e:
            logger.error(f"Signal analysis error: {e}", exc_info=True)
            return f"Error analyzing signals: {str(e)}"

    async def reflect_on_trades(self, lookback_days: int = 7) -> str:
        """
        Analyze historical trades and reflect on performance.
        
        Args:
            lookback_days: Number of days to analyze
            
        Returns:
            Reflection text
        """
        if not self.client:
            return "LLM client not available."
        
        try:
            # Fetch recent trades from backtest or live trading data
            cutoff_time = datetime.now() - timedelta(days=lookback_days)
            
            query = """
                SELECT symbol, ts_ms, signal_type, confidence, reason
                FROM signals
                WHERE ts_ms >= ? AND status = 'sent'
                ORDER BY ts_ms DESC
                LIMIT 50
            """
            
            result = self.db.conn.execute(query, (cutoff_time,))
            trades_df = result.df()
            
            if trades_df.empty:
                return f"No trading data in the last {lookback_days} days."
            
            # Prepare reflection prompt
            trades_text = self._format_dataframe_for_llm(trades_df)
            
            system_prompt = """You are a trading performance analyst. 
Analyze the provided trading history and provide insights on:
1. Win/loss patterns
2. Signal quality assessment
3. Areas for improvement
4. Market regime observations
5. Recommendations for strategy adjustment"""
            
            user_message = f"""Reflect on this trading history from the last {lookback_days} days:

{trades_text}

Provide a detailed analysis with specific improvement suggestions."""
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            reflection = response.choices[0].message.content
            
            self._log_insight("trade_reflection", reflection, trades_df)
            
            logger.info(f"✓ Trade reflection complete ({len(trades_df)} trades)")
            return reflection
        
        except Exception as e:
            logger.error(f"Trade reflection error: {e}", exc_info=True)
            return f"Error reflecting on trades: {str(e)}"

    async def analyze_anomalies(self, limit: int = 20) -> str:
        """
        Analyze recent anomaly events.
        
        Args:
            limit: Number of anomalies to analyze
            
        Returns:
            Analysis text
        """
        if not self.client:
            return "LLM client not available."
        
        try:
            query = """
                SELECT symbol, ts_ms, event_type, severity, confidence
                FROM anomaly_events
                ORDER BY ts_ms DESC
                LIMIT ?
            """
            
            result = self.db.conn.execute(query, (limit,))
            anomalies_df = result.df()
            
            if anomalies_df.empty:
                return "No anomalies detected."
            
            anomalies_text = self._format_dataframe_for_llm(anomalies_df)
            
            system_prompt = """You are an anomaly pattern analyst.
Identify and explain the anomalies in the provided data.
Assess their market significance and potential impact."""
            
            user_message = f"""Analyze these recent anomalies:

{anomalies_text}

What patterns do you observe? What is the market significance?"""
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            analysis = response.choices[0].message.content
            
            self._log_insight("anomaly_analysis", analysis, anomalies_df)
            
            logger.info(f"✓ Anomaly analysis complete ({len(anomalies_df)} anomalies)")
            return analysis
        
        except Exception as e:
            logger.error(f"Anomaly analysis error: {e}", exc_info=True)
            return f"Error analyzing anomalies: {str(e)}"

    async def chat(self, query: str, context: Optional[str] = None) -> str:
        """
        Natural language interaction with the agent.
        
        Args:
            query: User query
            context: Optional context (market data, etc.)
            
        Returns:
            Response text
        """
        if not self.client:
            return "LLM client not available. Agent running in read-only mode."
        
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            
            # Prepare context
            if not context:
                context = await self._get_market_context()
            
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": f"""You are OmerGPT, an AI-powered trading assistant.
You have access to:
- Real-time market data (prices, volumes, volatility)
- On-chain metrics (gas prices, transfers, inflows)
- AI-generated trading signals
- Anomaly detection results
- Historical performance data

Current Market Context:
{context}

Provide helpful, accurate, and actionable responses."""
                }
            ]
            
            # Add conversation history (last 5 exchanges)
            for msg in self.conversation_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.4,
                max_tokens=800
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.debug(f"✓ Chat response generated ({len(assistant_response)} chars)")
            return assistant_response
        
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            return f"Error processing query: {str(e)}"

    async def think_loop(self):
        """
        Autonomous thinking loop: periodic analysis and reflection.
        """
        self.running = True
        logger.info("✓ Autonomous thinking loop started")
        
        cycle = 0
        while self.running:
            try:
                cycle += 1
                logger.info(f"🧠 Agent cycle {cycle} starting...")
                
                # Cycle 1: Analyze signals
                if cycle % 3 == 1:
                    analysis = await self.analyze_signals(limit=5)
                    logger.info(f"Signal Analysis:\n{analysis}")
                
                # Cycle 2: Analyze anomalies
                elif cycle % 3 == 2:
                    analysis = await self.analyze_anomalies(limit=10)
                    logger.info(f"Anomaly Analysis:\n{analysis}")
                
                # Cycle 3: Reflect on trades
                else:
                    reflection = await self.reflect_on_trades(lookback_days=1)
                    logger.info(f"Trade Reflection:\n{reflection}")
                
                await asyncio.sleep(self.reflection_interval)
            
            except asyncio.CancelledError:
                logger.info("Thinking loop cancelled")
                break
            
            except Exception as e:
                logger.error(f"Think loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def stop(self):
        """Stop agent gracefully."""
        logger.info("Stopping agent...")
        self.running = False
        self._save_insights()

    # ==================== HELPER METHODS ====================

    async def _get_market_context(self) -> str:
        """Build current market context from database."""
        try:
            # Latest candles
            candles_query = """
                SELECT symbol, close, 
                       (close - LAG(close) OVER (PARTITION BY symbol ORDER BY ts_ms)) as change
                FROM candles
                WHERE ts_ms >= datetime('now', '-1 hour')
                ORDER BY ts_ms DESC
                LIMIT 10
            """
            
            candles = self.db.conn.execute(candles_query).df()
            
            # Recent signals
            signals_query = """
                SELECT symbol, signal_type, COUNT(*) as count
                FROM signals
                WHERE ts_ms >= datetime('now', '-1 hour')
                GROUP BY symbol, signal_type
            """
            
            signals = self.db.conn.execute(signals_query).df()
            
            # Recent anomalies
            anomalies_query = """
                SELECT symbol, COUNT(*) as count, MAX(confidence) as max_score
                FROM anomaly_events
                WHERE ts_ms >= datetime('now', '-1 hour')
                GROUP BY symbol
            """
            
            anomalies = self.db.conn.execute(anomalies_query).df()
            
            context = f"""Last Hour Summary:
Candles: {len(candles)} price updates
Signals: {len(signals)} signal groups
Anomalies: {len(anomalies)} anomaly groups"""
            
            return context
        
        except Exception as e:
            logger.warning(f"Context building error: {e}")
            return "Market context unavailable"

    def _format_dataframe_for_llm(self, df: pd.DataFrame, max_rows: int = 20) -> str:
        """Format DataFrame for LLM consumption."""
        if len(df) > max_rows:
            df = df.head(max_rows)
        
        return df.to_string(index=False)

    def _log_insight(self, insight_type: str, content: str, source_df: pd.DataFrame):
        """Log an insight."""
        insight = {
            "type": insight_type,
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "source_rows": len(source_df)
        }
        self.insights_log.append(insight)
        logger.debug(f"Insight logged: {insight_type}")

    def _save_insights(self):
        """Save insights to file."""
        try:
            if self.insights_log:
                filepath = f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filepath, "w") as f:
                    json.dump(self.insights_log, f, indent=2, default=str)
                logger.info(f"Insights saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save insights: {e}")


# ==================== INTERACTIVE CLI ====================

async def interactive_cli(agent: OmerGPTAgent):
    """Interactive CLI for agent interaction."""
    print("\n" + "="*70)
    print("🤖 OmerGPT Agent Interactive Mode")
    print("="*70)
    print("Commands:")
    print("  /signals    - Analyze recent signals")
    print("  /anomalies  - Analyze recent anomalies")
    print("  /trades     - Reflect on trading history")
    print("  /context    - Show market context")
    print("  /quit       - Exit")
    print("  (anything else) - Chat with agent")
    print("="*70 + "\n")
    
    loop = asyncio.get_event_loop()
    
    while True:
        try:
            # Get user input in non-blocking way
            user_input = await loop.run_in_executor(None, input, "You: ")
            
            if not user_input:
                continue
            
            if user_input == "/quit":
                print("👋 Goodbye!")
                break
            
            elif user_input == "/signals":
                print("\n📊 Analyzing signals...")
                result = await agent.analyze_signals()
                print(f"\n{result}\n")
            
            elif user_input == "/anomalies":
                print("\n⚠️ Analyzing anomalies...")
                result = await agent.analyze_anomalies()
                print(f"\n{result}\n")
            
            elif user_input == "/trades":
                print("\n📈 Reflecting on trades...")
                result = await agent.reflect_on_trades()
                print(f"\n{result}\n")
            
            elif user_input == "/context":
                context = await agent._get_market_context()
                print(f"\n📍 Market Context:\n{context}\n")
            
            else:
                print("\n🤔 Processing your query...")
                response = await agent.chat(user_input)
                print(f"\nAgent: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            logger.error(f"CLI error: {e}")
            print(f"\n❌ Error: {e}\n")


# ==================== MAIN EXECUTION ====================

async def main():
    """Main entry point."""
    import sys
    sys.path.insert(0, "src")
    
    from storage.db_manager import DatabaseManager
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("\n" + "="*70)
    print("🚀 OmerGPT Agent Startup")
    print("="*70 + "\n")
    
    # Initialize
    db = DatabaseManager("data/market_data.duckdb")
    agent = OmerGPTAgent(
        db_manager=db,
        model_name="gpt-4",
        reflection_interval=60
    )
    
    # Run interactive CLI
    await interactive_cli(agent)
    
    # Cleanup
    await agent.stop()
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
