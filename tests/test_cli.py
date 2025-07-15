import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console
from typer.testing import CliRunner

from redis_sizer.cli import _get_memory_usage, app


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

        # Setup the Lua script return value for _scan_and_measure_keys
        self.mock_script = MagicMock()
        self.mock_redis.register_script.return_value = self.mock_script
        # The script returns [cursor, keys, memory_values]
        self.mock_script.return_value = [
            0,  # cursor (0 means end of scan)
            ["test:key1", "test:key2", "other:key1"],  # keys
            [100, 200, 300],  # memory values
        ]

    def tearDown(self) -> None:
        self.mock_redis_patcher.stop()

    def test_analyze(self) -> None:
        # Execute the command
        result = self.runner.invoke(app, ["localhost"])

        # Verify the result
        self.assertEqual(result.exit_code, 0)
        self.assertIn("The total number of keys: 5", result.stdout)
        self.assertIn("Scanning and measuring keys...", result.stdout)
        self.assertIn("Memory Usage", result.stdout)
        self.assertIn("Took", result.stdout)

        # Verify Redis was called correctly
        self.mock_redis.dbsize.assert_called_once()
        self.mock_redis.register_script.assert_called()
        self.mock_redis.close.assert_called()


class TestGetMemoryUsage(unittest.TestCase):
    """Test the _get_memory_usage function."""

    def setUp(self) -> None:
        """Set up mock Redis client."""
        self.mock_redis = MagicMock()
        self.mock_script = MagicMock()
        self.mock_redis.register_script.return_value = self.mock_script
        self.mock_redis.dbsize.return_value = 100  # Total keys in database

    def test_scan_all_keys(self) -> None:
        """Test _get_memory_usage returns all keys when sample_size is None."""
        # Setup script to return [cursor, keys, memory_values]
        self.mock_script.return_value = [
            0,  # cursor (0 means end of scan)
            ["key1", "key2", "key3", "key4"],  # keys
            [100, 200, 300, 400],  # memory values
        ]

        # Call _get_memory_usage with sample_size=None to collect all keys
        memory_usage = _get_memory_usage(
            redis=self.mock_redis,
            pattern="*",
            batch_size=100,
            sample_size=None,
            total=100,
            console=Console(),
        )

        self.assertEqual(memory_usage, {"key1": 100, "key2": 200, "key3": 300, "key4": 400})
        self.mock_redis.register_script.assert_called_once()

    def test_scan_sample_keys(self) -> None:
        """Test _get_memory_usage stops scanning after reaching sample_size."""
        # Setup script to return [cursor, keys, memory_values]
        self.mock_script.return_value = [
            0,  # cursor (0 means end of scan)
            ["key1", "key2"],  # keys
            [100, 200],  # memory values
        ]

        # Specify sample_size so that only the first two keys should be returned
        memory_usage = _get_memory_usage(
            redis=self.mock_redis,
            pattern="*",
            batch_size=100,
            sample_size=2,
            total=100,
            console=Console(),
        )

        self.assertEqual(memory_usage, {"key1": 100, "key2": 200})

    def test_scan_no_keys(self) -> None:
        """Test _get_memory_usage returns empty when no keys are found."""
        # Setup script to return [cursor, keys, memory_values]
        self.mock_script.return_value = [
            0,  # cursor (0 means end of scan)
            [],  # no keys
            [],  # no memory values
        ]

        # Call _get_memory_usage with a pattern that doesn't match any keys
        memory_usage = _get_memory_usage(
            redis=self.mock_redis,
            pattern="nonexistent",
            batch_size=100,
            sample_size=None,
            total=100,
            console=Console(),
        )

        self.assertEqual(memory_usage, {})


if __name__ == "__main__":
    unittest.main()
