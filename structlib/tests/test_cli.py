
from click.testing import CliRunner

from structlib.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert result.output.startswith("Usage: cli [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...")