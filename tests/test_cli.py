import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console
from typer.testing import CliRunner

from redis_sizer.cli import (
    MemoryUnit,
    TableRow,
    _get_memory_unit_factor,
    _get_memory_usage,
    _parse_key_group,
    _print_memory_usage_table,
    _scan_keys,
    app,
)


class TestApp(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

        # Setup mocks
        self.mock_redis_patcher = patch("redis_sizer.cli.Redis")
        self.mock_redis_class = self.mock_redis_patcher.start()
        self.mock_redis = MagicMock()
        self.mock_redis_class.return_value = self.mock_redis

        # Configure the mock Redis instance
        self.mock_redis.dbsize.return_value = 5
        self.mock_redis.scan_iter.return_value = iter([b"test:key1", b"test:key2", b"other:key1"])

        # Setup the Lua script return value
        self.mock_script = MagicMock()
        self.mock_redis.register_script.return_value = self.mock_script
        self.mock_script.return_value = [100, 200, 300]  # Memory usage for each key

    def tearDown(self) -> None:
        self.mock_redis_patcher.stop()

    def test_invalid_namespace_level(self) -> None:
        result = self.runner.invoke(app, ["localhost", "--namespace-level", "-1"])
        self.assertNotEqual(result.exit_code, 0)

    def test_analyze(self) -> None:
        # Execute the command
        result = self.runner.invoke(app, ["localhost"])

        # Verify the result
        self.assertEqual(result.exit_code, 0)
        self.assertIn("The total number of keys: 5", result.stdout)
        self.assertIn("Scanning keys...", result.stdout)
        self.assertIn("Calculating memory usage...", result.stdout)
        self.assertIn("Took", result.stdout)

        # Verify Redis was called correctly
        self.mock_redis.dbsize.assert_called_once()
        self.mock_redis.scan_iter.assert_called_once()
        self.mock_redis.register_script.assert_called()
        self.mock_redis.close.assert_called()


class TestScanKeys(unittest.TestCase):
    """Test the _scan_keys function."""

    def test_scan_all_keys(self) -> None:
        """Test _scan_keys returns all keys when sample_size is None."""
        # Prepare a fake redis with a scan_iter method that yields bytes keys.
        fake_redis = MagicMock()
        fake_redis.scan_iter.return_value = iter([b"key1", b"key2", b"key3", b"key4"])

        # Call _scan_keys with sample_size=None to collect all keys.
        result: list[str] = _scan_keys(
            redis=fake_redis, pattern="*", count=100, sample_size=None, console=Console(), total=4
        )
        self.assertEqual(result, ["key1", "key2", "key3", "key4"])
        fake_redis.scan_iter.assert_called_once_with(match="*", count=100)

    def test_scan_sample_keys(self) -> None:
        """Test _scan_keys stops scanning after reaching sample_size."""
        # Prepare a fake redis with more keys than the sample size.
        fake_redis = MagicMock()
        fake_redis.scan_iter.return_value = iter([b"key1", b"key2", b"key3", b"key4", b"key5"])

        # Specify sample_size so that only the first two keys should be returned.
        result: list[str] = _scan_keys(
            redis=fake_redis, pattern="*", count=100, sample_size=2, console=Console(), total=5
        )
        self.assertEqual(result, ["key1", "key2"])
        fake_redis.scan_iter.assert_called_once_with(match="*", count=100)

    def test_scan_no_keys(self) -> None:
        """Test _scan_keys returns an empty list if no keys are yielded."""
        # Prepare a fake redis with no keys.
        fake_redis = MagicMock()
        fake_redis.scan_iter.return_value = iter([])

        # Call _scan_keys with a pattern that doesn't match any keys.
        result: list[str] = _scan_keys(
            redis=fake_redis, pattern="nonexistent", count=100, sample_size=None, console=Console()
        )
        self.assertEqual(result, [])
        fake_redis.scan_iter.assert_called_once_with(match="nonexistent", count=100)


class TestParseKeyGroup(unittest.TestCase):
    """Test the _parse_key_group function."""

    def test_parse_key_group_basic(self):
        """Test basic prefix parsing with default separator."""
        self.assertEqual(_parse_key_group("a:b:c", ":", 0), "a:b:c")
        self.assertEqual(_parse_key_group("a:b:c", ":", 1), "a")
        self.assertEqual(_parse_key_group("a:b:c", ":", 2), "a:b")
        self.assertEqual(_parse_key_group("a:b:c", ":", 3), "a:b:c")
        self.assertEqual(_parse_key_group("a:b:c", ":", 4), "a:b:c")

    def test_parse_key_group_custom_separator(self):
        """Test prefix parsing with custom separators"""
        self.assertEqual(_parse_key_group("a-b-c", "-", 0), "a-b-c")
        self.assertEqual(_parse_key_group("a-b-c", "-", 1), "a")
        self.assertEqual(_parse_key_group("a-b-c", "-", 2), "a-b")
        self.assertEqual(_parse_key_group("a-b-c", "-", 3), "a-b-c")
        self.assertEqual(_parse_key_group("a-b-c", "-", 4), "a-b-c")

    def test_parse_key_group_no_separator(self):
        """Test parsing when the separator doesn't exist in the key."""
        self.assertEqual(_parse_key_group("a", ":", 1), "a")
        self.assertEqual(_parse_key_group("a", ":", 2), "a")
        self.assertEqual(_parse_key_group("a", ":", 3), "a")

    def test_parse_key_group_empty_string(self):
        """Test parsing with an empty string as the key."""
        self.assertEqual(_parse_key_group("", ":", 1), "")
        self.assertEqual(_parse_key_group("", ":", 2), "")
        self.assertEqual(_parse_key_group("", "-", 3), "")

    def test_parse_key_group_consecutive_separators(self):
        """Test parsing with consecutive separators."""
        self.assertEqual(_parse_key_group("a::b:c", ":", 2), "a:")
        self.assertEqual(_parse_key_group(":a:b:c", ":", 2), ":a")
        self.assertEqual(_parse_key_group("a:b::", ":", 3), "a:b:")


class TestGetMemoryUsage(unittest.TestCase):
    """Test the _get_memory_usage function."""

    def setUp(self) -> None:
        """Set up mock Redis client."""
        self.mock_redis = MagicMock()
        self.mock_script = MagicMock()
        self.mock_redis.register_script.return_value = self.mock_script

    def test_get_memory_usage_basic(self) -> None:
        """Test basic functionality of _get_memory_usage."""
        # Setup
        keys = ["key1", "key2", "key3"]
        self.mock_script.return_value = [100, 200, 300]

        # Execute
        result = _get_memory_usage(self.mock_redis, keys)

        # Verify
        self.mock_redis.register_script.assert_called_once()
        self.mock_script.assert_called_once_with(keys=keys)
        self.assertEqual(result, {"key1": 100, "key2": 200, "key3": 300})

    def test_get_memory_usage_empty_keys(self) -> None:
        """Test _get_memory_usage with empty keys list."""
        # Setup
        keys = []
        self.mock_script.return_value = []

        # Execute
        result = _get_memory_usage(self.mock_redis, keys)

        # Verify
        self.mock_redis.register_script.assert_called_once()
        self.mock_script.assert_called_once_with(keys=keys)
        self.assertEqual(result, {})


class TestGetMemoryUnitFactor(unittest.TestCase):
    """Test the _get_memory_unit_factor function."""

    def test_memory_unit_factors(self) -> None:
        """Test the conversion factors for all memory units."""
        self.assertEqual(_get_memory_unit_factor(MemoryUnit.B), 1)
        self.assertEqual(_get_memory_unit_factor(MemoryUnit.KB), 1024)
        self.assertEqual(_get_memory_unit_factor(MemoryUnit.MB), 1024 * 1024)
        self.assertEqual(_get_memory_unit_factor(MemoryUnit.GB), 1024 * 1024 * 1024)

    def test_invalid_unit(self) -> None:
        """Test that an invalid memory unit raises a ValueError."""
        with self.assertRaises(ValueError):
            _get_memory_unit_factor("invalid_unit")  # type: ignore


class TestPrintMemoryUsage(unittest.TestCase):
    """Test the _print_memory_usage_table function."""

    def test_print_memory_usage(self) -> None:
        """
        Sanity check for _print_memory_usage_table.
        NOTE Verifying console output is flaky as results may vary based on the terminal width.
        """
        _print_memory_usage_table(
            title="Test Memory Usage",
            rows=[
                TableRow(
                    key="key",
                    count="1",
                    size="100",
                    avg_size="1",
                    min_size="1",
                    max_size="1",
                    percentage="100.00",
                )
            ],
            total_row=TableRow(
                key="1",
                count="1",
                size="100",
                avg_size="1",
                min_size="1",
                max_size="1",
                percentage="100.00",
            ),
            hidden_count=0,
            memory_unit=MemoryUnit.B,
            show_extra_stats=False,
            console=Console(),
        )


if __name__ == "__main__":
    unittest.main()
