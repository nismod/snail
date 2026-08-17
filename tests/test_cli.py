from snail.cli import snail


def test_cli_without_command_prints_help(capsys):
    snail([])

    captured = capsys.readouterr()
    assert "usage: snail" in captured.out
