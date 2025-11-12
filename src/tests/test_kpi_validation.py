# === AUTO PATCH START ===
# Auto-added test file to enforce KPIs
def test_ir_sharpe_thresholds():
    from src.validation import metrics
    ir, sharpe = metrics.get_latest_kpis()
    assert ir >= 1.0, f"IR below 1.0: {ir}"
    assert sharpe >= 1.5, f"Sharpe below 1.5: {sharpe}"
# === AUTO PATCH END ===
