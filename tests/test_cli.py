import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console
from typer.testing import CliRunner

from redis_sizer.cli import (
    MemoryUnit,
    TableRow,
    _build_key_tree,
    _generate_hierarchical_rows,
    _get_memory_unit_factor,
    _get_memory_usage,
    _print_hierarchical_table,
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

    def test_analyze(self) -> None:
        # Execute the command
        result = self.runner.invoke(app, ["localhost"])

        # Verify the result
        self.assertEqual(result.exit_code, 0)
        self.assertIn("The total number of keys: 5", result.stdout)
        self.assertIn("Scanning keys...", result.stdout)
        self.assertIn("Calculating memory usage...", result.stdout)
        self.assertIn("Memory Usage Hierarchical View", result.stdout)
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


class TestBuildKeyTree(unittest.TestCase):
    """Test the _build_key_tree function."""

    def test_build_key_tree_basic(self):
        """Test building a basic key tree."""
        keys = ["users:profiles:123", "users:profiles:456", "users:logs:789", "sessions:active:111"]
        memory_usage = {
            "users:profiles:123": 100,
            "users:profiles:456": 200,
            "users:logs:789": 300,
            "sessions:active:111": 400,
        }

        root = _build_key_tree(keys, memory_usage, ":")  # type: ignore

        # Check root structure
        self.assertEqual(len(root.children), 2)  # users and sessions
        self.assertIn("users", root.children)
        self.assertIn("sessions", root.children)

        # Check users branch
        users_node = root.children["users"]
        self.assertEqual(users_node.name, "users:")
        self.assertEqual(len(users_node.children), 2)  # profiles and logs
        self.assertEqual(users_node.size, 600)  # 100 + 200 + 300

        # Check profiles branch
        profiles_node = users_node.children["profiles"]
        self.assertEqual(profiles_node.name, "profiles:")
        self.assertEqual(len(profiles_node.keys), 2)
        self.assertEqual(profiles_node.size, 300)  # 100 + 200

    def test_build_key_tree_with_none_values(self):
        """Test building tree with some None memory values."""
        keys = ["a:b", "a:c", "d:e"]
        memory_usage = {"a:b": 100, "a:c": None, "d:e": 200}

        root = _build_key_tree(keys, memory_usage, ":")

        # a:c should be skipped
        self.assertEqual(root.size, 300)  # 100 + 200
        self.assertEqual(len(root.children["a"].keys), 1)  # Only a:b

    def test_build_key_tree_no_namespace(self):
        """Test building tree with keys that have no namespace."""
        keys = ["key1", "key2", "ns:key3"]
        memory_usage = {"key1": 100, "key2": 200, "ns:key3": 300}

        root = _build_key_tree(keys, memory_usage, ":")  # type: ignore

        # Root should have direct keys
        self.assertEqual(len(root.keys), 2)  # key1 and key2
        self.assertEqual(len(root.children), 1)  # ns
        self.assertEqual(root.size, 600)  # Total


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


class TestGenerateHierarchicalRows(unittest.TestCase):
    """Test the _generate_hierarchical_rows function."""

    def test_generate_hierarchical_rows(self):
        """Test generating rows from a key tree."""
        # Build a simple tree
        keys = ["users:profiles:123", "users:logs:456", "sessions:789"]
        memory_usage = {"users:profiles:123": 1000, "users:logs:456": 2000, "sessions:789": 3000}

        root = _build_key_tree(keys, memory_usage, ":")  # type: ignore
        rows, total_row = _generate_hierarchical_rows(root, memory_usage, MemoryUnit.B, None)  # type: ignore

        # Check that we have the right number of rows
        # Should have: sessions:, users:, profiles:, users:profiles:123, logs:, users:logs:456, sessions:789
        self.assertEqual(len(rows), 7)

        # Check total row
        self.assertEqual(total_row.count, "3")
        self.assertEqual(total_row.size, "6000")
        self.assertEqual(total_row.percentage, "100.00")

    def test_generate_hierarchical_rows_with_top(self):
        """Test generating rows with top limit."""
        # Build a tree with many keys
        keys = [f"ns:key{i}" for i in range(10)]
        memory_usage = {k: 100 * (i + 1) for i, k in enumerate(keys)}

        root = _build_key_tree(keys, memory_usage, ":")  # type: ignore
        rows, total_row = _generate_hierarchical_rows(root, memory_usage, MemoryUnit.B, top=3)  # type: ignore

        # Should have ns: row, top 3 keys, and "... more keys..." row
        found_more = any("... " in row.key and "more keys" in row.key for row in rows)
        self.assertTrue(found_more)


class TestPrintHierarchicalTable(unittest.TestCase):
    """Test the _print_hierarchical_table function."""

    def test_print_hierarchical_table(self) -> None:
        """
        Sanity check for _print_hierarchical_table.
        NOTE Verifying console output is flaky as results may vary based on the terminal width.
        """
        _print_hierarchical_table(
            title="Test Hierarchical Memory Usage",
            rows=[
                TableRow(
                    key="users:",
                    count="10",
                    size="1000",
                    avg_size="100",
                    min_size="50",
                    max_size="150",
                    percentage="50.00",
                    level=1,
                ),
                TableRow(
                    key="  profiles:",
                    count="5",
                    size="500",
                    avg_size="100",
                    min_size="80",
                    max_size="120",
                    percentage="25.00",
                    level=2,
                ),
            ],
            total_row=TableRow(
                key="Total Keys Scanned",
                count="20",
                size="2000",
                avg_size="100",
                min_size="50",
                max_size="150",
                percentage="100.00",
                level=0,
            ),
            memory_unit=MemoryUnit.B,
            console=Console(),
        )


if __name__ == "__main__":
    unittest.main()
