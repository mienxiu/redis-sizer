import unittest

from redis_sizer.models import KeyNode
from redis_sizer.tree_builder import KeyTreeBuilder


class TestKeyTreeBuilder(unittest.TestCase):
    """Test the KeyTreeBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.builder = KeyTreeBuilder()

    def test_build_tree_basic(self):
        """Test building a basic key tree."""
        memory_usage = {
            "users:profiles:123": 100,
            "users:profiles:456": 200,
            "users:logs:789": 300,
            "sessions:active:111": 400,
        }

        root = self.builder.build_tree(memory_usage, ":")

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

    def test_build_tree_no_namespace(self):
        """Test building tree with keys that have no namespace."""
        memory_usage = {"key1": 100, "key2": 200, "ns:key3": 300}

        root = self.builder.build_tree(memory_usage, ":")

        # Root should have direct keys
        self.assertEqual(len(root.keys), 2)  # key1 and key2
        self.assertEqual(len(root.children), 1)  # ns
        self.assertEqual(root.size, 600)  # Total
        self.assertIn("key1", root.keys)
        self.assertIn("key2", root.keys)

    def test_build_tree_empty(self):
        """Test building tree with empty memory usage."""
        root = self.builder.build_tree({}, ":")

        self.assertEqual(len(root.children), 0)
        self.assertEqual(len(root.keys), 0)
        self.assertEqual(root.size, 0)

    def test_build_tree_single_key(self):
        """Test building tree with a single key."""
        memory_usage = {"single:key": 500}

        root = self.builder.build_tree(memory_usage, ":")

        self.assertEqual(len(root.children), 1)
        self.assertIn("single", root.children)
        self.assertEqual(root.size, 500)

        single_node = root.children["single"]
        self.assertEqual(len(single_node.keys), 1)
        self.assertIn("single:key", single_node.keys)

    def test_build_tree_deep_nesting(self):
        """Test building tree with deeply nested keys."""
        memory_usage = {
            "a:b:c:d:e:key1": 100,
            "a:b:c:d:e:key2": 200,
            "a:b:c:d:f:key3": 300,
        }

        root = self.builder.build_tree(memory_usage, ":")

        # Navigate through the tree
        self.assertIn("a", root.children)
        a_node = root.children["a"]
        self.assertIn("b", a_node.children)
        b_node = a_node.children["b"]
        self.assertIn("c", b_node.children)
        c_node = b_node.children["c"]
        self.assertIn("d", c_node.children)
        d_node = c_node.children["d"]
        self.assertEqual(len(d_node.children), 2)  # e and f
        self.assertEqual(root.size, 600)

    def test_custom_separator(self):
        """Test building tree with custom separator."""
        builder = KeyTreeBuilder()
        memory_usage = {
            "users/profiles/123": 100,
            "users/logs/456": 200,
        }

        root = builder.build_tree(memory_usage, "/")

        self.assertIn("users", root.children)
        users_node = root.children["users"]
        self.assertEqual(users_node.name, "users/")
        self.assertEqual(len(users_node.children), 2)

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
        self.builder._propagate_sizes(root)

        # Check sizes after propagation
        self.assertEqual(grandchild.size, 100)
        self.assertEqual(child1.size, 300)  # 100 + 200
        self.assertEqual(child2.size, 700)  # 300 + 400
        self.assertEqual(root.size, 1000)  # 300 + 700

        # Check that all sizes are propagated correctly
        self.assertEqual(len(root.sizes), 4)  # All 4 sizes
        self.assertEqual(len(child1.sizes), 2)  # grandchild + child1's own
        self.assertEqual(len(grandchild.sizes), 1)  # Only its own


if __name__ == "__main__":
    unittest.main()
