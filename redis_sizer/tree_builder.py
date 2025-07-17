from redis_sizer.models import KeyNode


class KeyTreeBuilder:
    """
    Build and manage hierarchical tree structures from Redis key patterns.
    """

    def build_tree(self, memory_usage: dict[str, int], separator: str = ":") -> KeyNode:
        """
        Build a hierarchical tree structure from the memory usage dictionary.

        Args:
            memory_usage: Dictionary mapping Redis keys to their memory usage
            separator: The character(s) used to separate namespaces in keys

        Returns:
            Root node of the constructed tree
        """
        root = KeyNode(name="", full_path="", level=0, keys=[], size=0, sizes=[], children={})

        for key, size in memory_usage.items():
            parts = key.split(separator)
            current_node = root
            current_path = ""

            # Navigate/create path in tree
            for i, part in enumerate(parts[:-1]):
                current_path = separator.join(parts[: i + 1]) + separator
                if part not in current_node.children:
                    current_node.children[part] = KeyNode(
                        name=part + separator,
                        full_path=current_path,
                        level=i + 1,
                        keys=[],
                        size=0,
                        sizes=[],
                        children={},
                    )
                current_node = current_node.children[part]

            # Add key to the appropriate level
            if len(parts) > 1:
                # Key has namespace, add to parent
                current_node.keys.append(key)
                current_node.size += size
                current_node.sizes.append(size)
            else:
                # Key has no namespace, add to root
                root.keys.append(key)
                root.size += size
                root.sizes.append(size)

        # Propagate sizes up the tree
        self._propagate_sizes(root)

        return root

    def _propagate_sizes(self, node: KeyNode) -> tuple[int, list[int]]:
        """
        Recursively propagate sizes from children to parents.

        Args:
            node: The node to process

        Returns:
            Tuple of (total_size, all_sizes) for this subtree
        """
        total_size = node.size
        all_sizes = node.sizes.copy()

        for child in node.children.values():
            child_size, child_sizes = self._propagate_sizes(child)
            total_size += child_size
            all_sizes.extend(child_sizes)

        node.size = total_size
        node.sizes = all_sizes
        return total_size, all_sizes
