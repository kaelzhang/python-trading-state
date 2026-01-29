import pytest

from trading_state.analyzer.types import AnalyzerType


def test_analyzer_types():
    assert str(AnalyzerType.TOTAL_RETURN) == 'Total Return'
    assert repr(AnalyzerType.TOTAL_RETURN) == 'TOTAL_RETURN'
    assert 'overall change' in AnalyzerType.TOTAL_RETURN.description

    with pytest.raises(ValueError, match='No parameters'):
        AnalyzerType.TOTAL_RETURN.params(trading_days=365)
