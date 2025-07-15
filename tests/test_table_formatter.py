import unittest

from redis_sizer.models import MemoryUnit, TableRow
from redis_sizer.table_formatter import TableFormatter
from redis_sizer.tree_builder import KeyTreeBuilder


class TestTableFormatter(unittest.TestCase):
    """Test the TableFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = TableFormatter()
        self.builder = KeyTreeBuilder()

    def test_generate_rows_basic(self):
        """Test generating rows from a key tree."""
        # Build a simple tree
        memory_usage = {
            "users:profiles:123": 1000,
            "users:logs:456": 2000,
            "sessions:789": 3000,
        }

        root = self.builder.build_tree(memory_usage, ":")
        rows, total_row = self.formatter.generate_rows(root, memory_usage, MemoryUnit.B, None)

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
        memory_usage = {f"ns:key{i}": 100 * (i + 1) for i in range(10)}

        root = self.builder.build_tree(memory_usage, ":")
        rows, total_row = self.formatter.generate_rows(
            root, memory_usage, MemoryUnit.B, max_leaves=3
        )

        # Should have "... more keys ..." row
        found_more = any("... " in row.key and "more keys" in row.key for row in rows)
        self.assertTrue(found_more)

        # Count actual key rows (not namespace or ... rows)
        key_rows = [
            row for row in rows if not row.key.strip().endswith(":") and "..." not in row.key
        ]
        self.assertEqual(len(key_rows), 3)  # Should have exactly max_leaves 3 keys

    def test_generate_rows_memory_units(self):
        """Test generating rows with different memory units."""
        memory_usage = {"test:key": 1536}  # 1.5 KB

        root = self.builder.build_tree(memory_usage, ":")

        # Test with KB
        rows, total_row = self.formatter.generate_rows(root, memory_usage, MemoryUnit.KB, None)
        self.assertEqual(total_row.size, "1.50")
        self.assertEqual(total_row.avg_size, "1.5000")

        # Test with MB
        rows, total_row = self.formatter.generate_rows(root, memory_usage, MemoryUnit.MB, None)
        self.assertEqual(total_row.size, "0.00")  # 1536 bytes is ~0.0015 MB

        # Test with B (integer display)
        rows, total_row = self.formatter.generate_rows(root, memory_usage, MemoryUnit.B, None)
        self.assertEqual(total_row.size, "1536")
        self.assertEqual(total_row.avg_size, "1536")

    def test_generate_rows_empty_tree(self):
        """Test generating rows from empty tree."""
        root = self.builder.build_tree({}, ":")
        rows, total_row = self.formatter.generate_rows(root, {}, MemoryUnit.B, None)

        self.assertEqual(len(rows), 0)
        self.assertEqual(total_row.count, "0")
        self.assertEqual(total_row.size, "0")

    def test_generate_rows_statistics(self):
        """Test that statistics (min, max, avg) are calculated correctly."""
        memory_usage = {
            "ns:key1": 100,
            "ns:key2": 200,
            "ns:key3": 300,
        }

        root = self.builder.build_tree(memory_usage, ":")
        rows, total_row = self.formatter.generate_rows(root, memory_usage, MemoryUnit.B, None)

        # Find the namespace row
        ns_row = next(row for row in rows if row.key.strip() == "ns:")

        self.assertEqual(ns_row.count, "3")
        self.assertEqual(ns_row.size, "600")
        self.assertEqual(ns_row.avg_size, "200")
        self.assertEqual(ns_row.min_size, "100")
        self.assertEqual(ns_row.max_size, "300")

    def test_generate_table_basic(self):
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

        table = self.formatter.generate_table(
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

    def test_generate_table_different_memory_units(self):
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
        table_kb = self.formatter.generate_table(
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
        table_mb = self.formatter.generate_table(
            title="Test MB",
            rows=rows,
            total_row=total_row,
            memory_unit=MemoryUnit.MB,
        )

        column_headers = [col.header for col in table_mb.columns]
        self.assertIn("Size (MB)", column_headers)
        self.assertIn("Avg Size (MB)", column_headers)

        # Test with GB unit
        table_gb = self.formatter.generate_table(
            title="Test GB",
            rows=rows,
            total_row=total_row,
            memory_unit=MemoryUnit.GB,
        )

        column_headers = [col.header for col in table_gb.columns]
        self.assertIn("Size (GB)", column_headers)

    def test_generate_rows_truncation_styling(self):
        """Test that truncation rows have proper styling flag."""
        memory_usage = {f"ns:key{i}": 100 for i in range(10)}

        root = self.builder.build_tree(memory_usage, ":")
        rows, _ = self.formatter.generate_rows(root, memory_usage, MemoryUnit.B, max_leaves=3)

        # Find truncation row
        truncation_rows = [row for row in rows if row.is_truncation_row]
        self.assertEqual(len(truncation_rows), 1)
        self.assertIn("... 7 more keys ...", truncation_rows[0].key)

    def test_zero_size_keys_filtered(self):
        """Test that zero-size keys are filtered out."""
        memory_usage = {
            "ns:key1": 100,
            "ns:key2": 0,  # Should be filtered
            "ns:key3": 200,
        }

        root = self.builder.build_tree(memory_usage, ":")
        rows, total_row = self.formatter.generate_rows(root, memory_usage, MemoryUnit.B, None)

        # Count leaf key rows
        leaf_rows = [
            row
            for row in rows
            if not row.key.strip().endswith(":")
            and not row.is_truncation_row
            and row.key.strip().startswith("ns:key")
        ]
        self.assertEqual(len(leaf_rows), 2)  # Only 2 non-zero keys


if __name__ == "__main__":
    unittest.main()
