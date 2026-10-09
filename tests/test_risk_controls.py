from share.backtest import risk_controls


def test_warning_drawdown_blocks_entries_without_forcing_liquidation():
    block_entries, force_liquidation = risk_controls(-0.06, False)

    assert block_entries is True
    assert force_liquidation is False


def test_hard_drawdown_forces_liquidation():
    block_entries, force_liquidation = risk_controls(-0.09, False)

    assert block_entries is True
    assert force_liquidation is True


def test_active_cooldown_blocks_entries_and_forces_liquidation():
    block_entries, force_liquidation = risk_controls(-0.01, True)

    assert block_entries is True
    assert force_liquidation is True
