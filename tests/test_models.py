import unittest

from redis_sizer.models import KeyNode, MemoryUnit, TableRow


class TestMemoryUnit(unittest.TestCase):
    """Test the MemoryUnit enum."""

    def test_memory_unit_factors(self):
        """Test the conversion factors for all memory units."""
        self.assertEqual(MemoryUnit.B.get_factor(), 1)
        self.assertEqual(MemoryUnit.KB.get_factor(), 1024)
        self.assertEqual(MemoryUnit.MB.get_factor(), 1024 * 1024)
        self.assertEqual(MemoryUnit.GB.get_factor(), 1024 * 1024 * 1024)

    def test_memory_unit_string_values(self):
        """Test that memory units have correct string values."""
        self.assertEqual(MemoryUnit.B.value, "B")
        self.assertEqual(MemoryUnit.KB.value, "KB")
        self.assertEqual(MemoryUnit.MB.value, "MB")
        self.assertEqual(MemoryUnit.GB.value, "GB")

    def test_memory_unit_upper(self):
        """Test that memory units can be uppercased."""
        self.assertEqual(MemoryUnit.B.upper(), "B")
        self.assertEqual(MemoryUnit.KB.upper(), "KB")
        self.assertEqual(MemoryUnit.MB.upper(), "MB")
        self.assertEqual(MemoryUnit.GB.upper(), "GB")


class TestTableRow(unittest.TestCase):
    """Test the TableRow dataclass."""

    def test_table_row_creation(self):
        """Test creating a TableRow with all fields."""
        row = TableRow(
            key="test:key",
            count="10",
            size="1000",
            avg_size="100",
            min_size="50",
            max_size="150",
            percentage="25.00",
            level=2,
            is_truncation_row=False,
        )

        self.assertEqual(row.key, "test:key")
        self.assertEqual(row.count, "10")
        self.assertEqual(row.size, "1000")
        self.assertEqual(row.avg_size, "100")
        self.assertEqual(row.min_size, "50")
        self.assertEqual(row.max_size, "150")
        self.assertEqual(row.percentage, "25.00")
        self.assertEqual(row.level, 2)
        self.assertFalse(row.is_truncation_row)

    def test_table_row_defaults(self):
        """Test TableRow default values."""
        row = TableRow(
            key="test",
            count="1",
            size="100",
            avg_size="100",
            min_size="100",
            max_size="100",
            percentage="100",
        )

        self.assertEqual(row.level, 0)  # Default level
        self.assertFalse(row.is_truncation_row)  # Default is_truncation_row

    def test_table_row_truncation(self):
        """Test creating a truncation row."""
        row = TableRow(
            key="... 5 more keys ...",
            count="",
            size="",
            avg_size="",
            min_size="",
            max_size="",
            percentage="",
            level=1,
            is_truncation_row=True,
        )

        self.assertTrue(row.is_truncation_row)
        self.assertEqual(row.count, "")
        self.assertEqual(row.size, "")


class TestKeyNode(unittest.TestCase):
    """Test the KeyNode dataclass."""

    def test_key_node_creation(self):
        """Test creating a KeyNode with all fields."""
        node = KeyNode(
            name="users:",
            full_path="users:",
            level=1,
            keys=["users:key1", "users:key2"],
            size=500,
            sizes=[200, 300],
            children={},
        )

        self.assertEqual(node.name, "users:")
        self.assertEqual(node.full_path, "users:")
        self.assertEqual(node.level, 1)
        self.assertEqual(node.keys, ["users:key1", "users:key2"])
        self.assertEqual(node.size, 500)
        self.assertEqual(node.sizes, [200, 300])
        self.assertEqual(node.children, {})

    def test_key_node_with_children(self):
        """Test creating KeyNode with children."""
        child_node = KeyNode(
            name="profiles:",
            full_path="users:profiles:",
            level=2,
            keys=["users:profiles:123"],
            size=100,
            sizes=[100],
            children={},
        )

        parent_node = KeyNode(
            name="users:",
            full_path="users:",
            level=1,
            keys=[],
            size=100,
            sizes=[100],
            children={"profiles": child_node},
        )

        self.assertIn("profiles", parent_node.children)
        self.assertEqual(parent_node.children["profiles"], child_node)
        self.assertEqual(parent_node.children["profiles"].level, 2)

    def test_key_node_root(self):
        """Test creating a root KeyNode."""
        root = KeyNode(
            name="",
            full_path="",
            level=0,
            keys=[],
            size=0,
            sizes=[],
            children={},
        )

        self.assertEqual(root.name, "")
        self.assertEqual(root.full_path, "")
        self.assertEqual(root.level, 0)
        self.assertEqual(len(root.keys), 0)
        self.assertEqual(root.size, 0)

    def test_key_node_modification(self):
        """Test that KeyNode fields can be modified."""
        node = KeyNode(
            name="test:",
            full_path="test:",
            level=1,
            keys=[],
            size=0,
            sizes=[],
            children={},
        )

        # Modify fields
        node.keys.append("test:key1")
        node.size = 100
        node.sizes.append(100)

        self.assertEqual(len(node.keys), 1)
        self.assertEqual(node.size, 100)
        self.assertEqual(node.sizes, [100])


if __name__ == "__main__":
    unittest.main()
