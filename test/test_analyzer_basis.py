import pytest

from trading_state.analyzer.types import AnalyzerType


def test_analyzer_types():
    assert str(AnalyzerType.TOTAL_RETURN) == 'Total Return'
    assert repr(AnalyzerType.TOTAL_RETURN) == 'TOTAL_RETURN'
    assert 'overall change' in AnalyzerType.TOTAL_RETURN.description

    with pytest.raises(ValueError, match='No parameters'):
        AnalyzerType.TOTAL_RETURN.params(trading_days=365)


def test_analyzer_type_params():
    analyzer, params = AnalyzerType.SHARPE_RATIO.params(trading_days=365, risk_free_rate=0.01)
    assert analyzer is AnalyzerType.SHARPE_RATIO
    assert params.trading_days == 365
    assert params.risk_free_rate == 0.01
