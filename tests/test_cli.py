import unittest
from unittest.mock import MagicMock, patch

from rich.console import Console
from typer.testing import CliRunner

from redis_sizer.cli import (
    KeyNode,
    MemoryUnit,
    TableRow,
    _build_key_tree,
    _generate_rows,
    _generate_table,
    _get_memory_usage,
    _propagate_sizes,
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


class TestBuildKeyTree(unittest.TestCase):
    """Test the _build_key_tree function."""

    def test_build_key_tree_basic(self):
        """Test building a basic key tree."""
        memory_usage: dict[str, int] = {
            "users:profiles:123": 100,
            "users:profiles:456": 200,
            "users:logs:789": 300,
            "sessions:active:111": 400,
        }

        root = _build_key_tree(memory_usage, ":")

        # Check root structure
        self.assertEqual(len(root.children), 2)  # users and sessions
        self.assertIn("users", root.children)
        self.assertIn("sessions", root.children)
        self.assertEqual(root.size, 1000)  # Total size

        # Check users branch
        users_node = root.children["users"]
        self.assertEqual(users_node.name, "users:")
        self.assertEqual(users_node.level, 1)
        self.assertEqual(len(users_node.children), 2)  # profiles and logs
        self.assertEqual(users_node.size, 600)  # 100 + 200 + 300

        # Check profiles branch
        profiles_node = users_node.children["profiles"]
        self.assertEqual(profiles_node.name, "profiles:")
        self.assertEqual(profiles_node.level, 2)
        self.assertEqual(len(profiles_node.keys), 2)
        self.assertEqual(profiles_node.size, 300)  # 100 + 200
        self.assertIn("users:profiles:123", profiles_node.keys)
        self.assertIn("users:profiles:456", profiles_node.keys)

    def test_build_key_tree_no_namespace(self):
        """Test building tree with keys that have no namespace."""
        memory_usage: dict[str, int] = {"key1": 100, "key2": 200, "ns:key3": 300}

        root = _build_key_tree(memory_usage, ":")

        # Root should have direct keys
        self.assertEqual(len(root.keys), 2)  # key1 and key2
        self.assertEqual(len(root.children), 1)  # ns
        self.assertEqual(root.size, 600)  # Total
        self.assertIn("key1", root.keys)
        self.assertIn("key2", root.keys)


class TestPropagateSizes(unittest.TestCase):
    """Test the _propagate_sizes function."""

    def test_propagate_sizes(self):
        """Test that sizes are properly propagated up the tree."""
        # Create a simple tree structure manually
        root = KeyNode(name="", full_path="", level=0, keys=[], size=0, sizes=[], children={})
        child1 = KeyNode(
            name="child1:", full_path="child1", level=1, keys=[], size=0, sizes=[], children={}
        )
        child2 = KeyNode(
            name="child2:", full_path="child2", level=1, keys=[], size=0, sizes=[], children={}
        )
        grandchild = KeyNode(
            name="gc:", full_path="child1:gc", level=2, keys=[], size=0, sizes=[], children={}
        )

        # Set up relationships
        root.children = {"child1": child1, "child2": child2}
        child1.children = {"gc": grandchild}

        # Add some keys with sizes
        grandchild.keys = ["child1:gc:key1"]
        grandchild.sizes = [100]
        grandchild.size = 100  # Initial size
        child1.keys = ["child1:key1"]
        child1.sizes = [200]
        child1.size = 200  # Initial size
        child2.keys = ["child2:key1", "child2:key2"]
        child2.sizes = [300, 400]
        child2.size = 700  # Initial size

        # Propagate sizes
        _propagate_sizes(root)

        # Check sizes after propagation
        self.assertEqual(grandchild.size, 100)
        self.assertEqual(child1.size, 300)  # 100 + 200
        self.assertEqual(child2.size, 700)  # 300 + 400
        self.assertEqual(root.size, 1000)  # 300 + 700


class TestGetMemoryUnitFactor(unittest.TestCase):
    """Test the MemoryUnit.get_factor method."""

    def test_memory_unit_factors(self) -> None:
        """Test the conversion factors for all memory units."""
        self.assertEqual(MemoryUnit.B.get_factor(), 1)
        self.assertEqual(MemoryUnit.KB.get_factor(), 1024)
        self.assertEqual(MemoryUnit.MB.get_factor(), 1024 * 1024)
        self.assertEqual(MemoryUnit.GB.get_factor(), 1024 * 1024 * 1024)


class TestGenerateeRows(unittest.TestCase):
    """Test the _generate_rows function."""

    def test_generate_rows(self):
        """Test generating rows from a key tree."""
        # Build a simple tree
        memory_usage: dict[str, int] = {
            "users:profiles:123": 1000,
            "users:logs:456": 2000,
            "sessions:789": 3000,
        }

        root = _build_key_tree(memory_usage, ":")
        rows, total_row = _generate_rows(root, memory_usage, MemoryUnit.B, None)

        # Check that we have the right structure
        # Should have hierarchical rows with proper indentation
        key_texts = [row.key.strip() for row in rows]

        # Check for namespace rows
        self.assertIn("sessions:", key_texts)
        self.assertIn("users:", key_texts)

        # Check for indented rows (they should have spaces in the key field)
        indented_rows = [row for row in rows if row.key.startswith("  ")]
        self.assertGreater(len(indented_rows), 0)

        # Check total row
        self.assertEqual(total_row.count, "3")
        self.assertEqual(total_row.size, "6000")
        self.assertEqual(total_row.percentage, "100.00")

    def test_generate_rows_with_max_leaves(self):
        """Test generating rows with max_leaves limit."""
        # Build a tree with many keys
        memory_usage: dict[str, int] = {f"ns:key{i}": 100 * (i + 1) for i in range(10)}

        root = _build_key_tree(memory_usage, ":")
        rows, total_row = _generate_rows(root, memory_usage, MemoryUnit.B, max_leaves=3)

        # Should have "... more keys ..." row
        found_more = any("... " in row.key and "more keys" in row.key for row in rows)
        self.assertTrue(found_more)

        # Count actual key rows (not namespace or ... rows)
        key_rows = [
            row for row in rows if not row.key.strip().endswith(":") and "..." not in row.key
        ]
        self.assertEqual(len(key_rows), 3)  # Should have exactly max_leaves 3 keys


class TestGenerateTable(unittest.TestCase):
    """Test the _generate_table function."""

    def test_generate_table_basic(self) -> None:
        """Test basic table generation with simple rows."""
        rows = [
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
        ]

        total_row = TableRow(
            key="Total Keys Scanned",
            count="20",
            size="2000",
            avg_size="100",
            min_size="50",
            max_size="150",
            percentage="100.00",
            level=0,
        )

        table = _generate_table(
            title="Test Memory Usage",
            rows=rows,
            total_row=total_row,
            memory_unit=MemoryUnit.B,
        )

        # Check that table is created successfully
        self.assertIsNotNone(table)
        self.assertEqual(table.title, "Test Memory Usage")
        self.assertEqual(len(table.columns), 7)  # 7 columns expected

        # Check column headers
        column_headers = [col.header for col in table.columns]
        self.assertIn("Key", column_headers)
        self.assertIn("Count", column_headers)
        self.assertIn("Size (B)", column_headers)
        self.assertIn("Avg Size (B)", column_headers)
        self.assertIn("Min Size (B)", column_headers)
        self.assertIn("Max Size (B)", column_headers)
        self.assertIn("Memory Usage (%)", column_headers)

    def test_generate_table_different_memory_units(self) -> None:
        """Test table generation with different memory units."""
        rows = [
            TableRow(
                key="test:key",
                count="1",
                size="1.50",
                avg_size="1.50",
                min_size="1.50",
                max_size="1.50",
                percentage="100.00",
                level=1,
            ),
        ]

        total_row = TableRow(
            key="Total Keys Scanned",
            count="1",
            size="1.50",
            avg_size="1.50",
            min_size="1.50",
            max_size="1.50",
            percentage="100.00",
            level=0,
        )

        # Test with KB unit
        table_kb = _generate_table(
            title="Test KB",
            rows=rows,
            total_row=total_row,
            memory_unit=MemoryUnit.KB,
        )

        column_headers = [col.header for col in table_kb.columns]
        self.assertIn("Size (KB)", column_headers)
        self.assertIn("Avg Size (KB)", column_headers)
        self.assertIn("Min Size (KB)", column_headers)
        self.assertIn("Max Size (KB)", column_headers)

        # Test with MB unit
        table_mb = _generate_table(
            title="Test MB",
            rows=rows,
            total_row=total_row,
            memory_unit=MemoryUnit.MB,
        )

        column_headers = [col.header for col in table_mb.columns]
        self.assertIn("Size (MB)", column_headers)
        self.assertIn("Avg Size (MB)", column_headers)

        # Test with GB unit
        table_gb = _generate_table(
            title="Test GB",
            rows=rows,
            total_row=total_row,
            memory_unit=MemoryUnit.GB,
        )

        column_headers = [col.header for col in table_gb.columns]
        self.assertIn("Size (GB)", column_headers)


if __name__ == "__main__":
    unittest.main()
