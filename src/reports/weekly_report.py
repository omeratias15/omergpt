"""
Weekly Report Generator for omerGPT
Run weekly to generate comprehensive reports
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.reports.report_generator import omerGPTReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def send_telegram_report(report_path: str):
    """Send report via Telegram"""
    try:
        from src.auto_recovery.error_notifier import get_notifier
        notifier = get_notifier()

        message = f"📈 Weekly Report Generated\n\nReport saved to:\n{Path(report_path).name}"
        await notifier.send_message(message, priority="INFO")

    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")

async def generate_weekly_report():
    """Generate weekly trading report"""
    logger.info("=" * 60)
    logger.info("Starting Weekly Report Generation")
    logger.info("=" * 60)

    try:
        generator = omerGPTReportGenerator()

        # Load data for 7 days
        logger.info("Loading trading metrics (7 days)...")
        metrics = generator.load_trading_metrics(days=7)

        logger.info("Loading sentiment data...")
        sentiment = generator.load_sentiment_data()

        logger.info("Loading macro regime...")
        macro = generator.load_macro_regime()

        # Generate chart
        logger.info("Creating performance charts...")
        chart_filename = f"weekly_chart_{datetime.now().strftime('%Y%m%d')}.png"
        chart_path = generator.create_performance_chart(metrics, chart_filename)

        # Generate reports
        logger.info("Generating HTML report...")
        html_path = generator.generate_html_report(metrics, sentiment, macro, period="Weekly")

        logger.info("Generating PDF report...")
        pdf_path = generator.generate_pdf_report(metrics, sentiment, macro, chart_path, period="Weekly")

        logger.info("=" * 60)
        logger.info("Weekly Report Complete")
        logger.info(f"HTML: {html_path}")
        logger.info(f"PDF: {pdf_path}")
        logger.info("=" * 60)

        # Send notification
        await send_telegram_report(pdf_path)

        return {"html": html_path, "pdf": pdf_path}

    except Exception as e:
        logger.error(f"Error generating weekly report: {e}", exc_info=True)
        raise

def main():
    """Main entry point"""
    try:
        result = asyncio.run(generate_weekly_report())
        print("\n✅ Weekly report generated successfully!")
        print(f"HTML: {result['html']}")
        print(f"PDF: {result['pdf']}")
    except KeyboardInterrupt:
        logger.info("Report generation cancelled by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
