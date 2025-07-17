from rich import box
from rich.table import Table

from redis_sizer.models import KeyNode, MemoryUnit, TableRow


class TableFormatter:
    """
    Format hierarchical key data into Rich tables for display.
    """

    def generate_rows(
        self,
        root: KeyNode,
        memory_usage: dict[str, int],
        memory_unit: MemoryUnit,
        max_leaves: int | None = None,
    ) -> tuple[list[TableRow], TableRow]:
        """
        Generate table rows from the tree structure with proper indentation.

        Args:
            root: Root node of the key tree
            memory_usage: Dictionary mapping keys to their memory usage
            memory_unit: Unit for memory display
            max_leaves: Maximum number of leaf keys to display per namespace

        Returns:
            Tuple of (rows, total_row)
        """
        rows: list[TableRow] = []
        factor = memory_unit.get_factor()

        # Track overall statistics
        overall_min: int | None = None
        overall_max: int = 0
        total_count: int = 0
        total_size: int = 0

        def traverse_node(node: KeyNode, parent_size: int, is_last_level: bool = False):
            nonlocal overall_min, overall_max, total_count, total_size

            # Update overall stats
            if node.sizes:
                for size in node.sizes:
                    overall_min = size if overall_min is None else min(overall_min, size)
                    overall_max = max(overall_max, size)
                total_count += len(node.keys)

            # Handle direct keys at this level
            if node.keys:
                # Sort keys by size
                key_sizes = [(k, memory_usage[k]) for k in node.keys]
                key_sizes.sort(key=lambda x: 0 if x[1] is None else x[1], reverse=True)

                # Apply max_leaves limit only at leaf level
                if is_last_level and max_leaves and len(key_sizes) > max_leaves:
                    displayed_keys = key_sizes[:max_leaves]
                    hidden_count = len(key_sizes) - max_leaves
                else:
                    displayed_keys = key_sizes
                    hidden_count = 0

                # Add rows for direct keys
                for key, size in displayed_keys:
                    if size == 0:
                        continue

                    display_key = "  " * node.level + key
                    size_converted = (size or 0) / factor
                    display_size = (
                        f"{size_converted:.2f}"
                        if memory_unit != MemoryUnit.B
                        else f"{int(size_converted)}"
                    )
                    percentage = ((size or 0) / parent_size * 100) if parent_size else 0

                    rows.append(
                        TableRow(
                            key=display_key,
                            count="1",
                            size=display_size,
                            avg_size="",
                            min_size="",
                            max_size="",
                            percentage=f"{percentage:.2f}",
                            level=node.level,
                        )
                    )

                if hidden_count > 0:
                    rows.append(
                        TableRow(
                            key="  " * node.level + f"... {hidden_count} more keys ...",
                            count="",
                            size="",
                            avg_size="",
                            min_size="",
                            max_size="",
                            percentage="",
                            level=node.level,
                            is_truncation_row=True,
                        )
                    )

            # Process children namespaces
            if node.children:
                # Sort children by size
                sorted_children = sorted(
                    node.children.items(), key=lambda x: x[1].size, reverse=True
                )

                for child_name, child_node in sorted_children:
                    if child_node.size == 0:
                        continue

                    # Add namespace row
                    display_key = "  " * (child_node.level - 1) + child_node.name
                    size_converted = child_node.size / factor
                    display_size = (
                        f"{size_converted:.2f}"
                        if memory_unit != MemoryUnit.B
                        else f"{int(size_converted)}"
                    )

                    avg_size = (
                        (sum(child_node.sizes) / len(child_node.sizes)) / factor
                        if child_node.sizes
                        else 0
                    )
                    min_size = min(child_node.sizes) / factor if child_node.sizes else 0
                    max_size = max(child_node.sizes) / factor if child_node.sizes else 0
                    percentage = (child_node.size / parent_size * 100) if parent_size else 0

                    # Calculate total count recursively
                    def count_keys(n: KeyNode) -> int:
                        count = len(n.keys)
                        for child in n.children.values():
                            count += count_keys(child)
                        return count

                    rows.append(
                        TableRow(
                            key=display_key,
                            count=str(count_keys(child_node)),
                            size=display_size,
                            avg_size=f"{avg_size:.4f}"
                            if memory_unit != MemoryUnit.B
                            else f"{int(avg_size)}",
                            min_size=f"{min_size:.4f}"
                            if memory_unit != MemoryUnit.B
                            else f"{int(min_size)}",
                            max_size=f"{max_size:.4f}"
                            if memory_unit != MemoryUnit.B
                            else f"{int(max_size)}",
                            percentage=f"{percentage:.2f}",
                            level=child_node.level,
                        )
                    )

                    # Check if this child has no sub-children (is last level)
                    child_is_last = len(child_node.children) == 0
                    traverse_node(child_node, parent_size, child_is_last)

        # Start traversal
        total_size = root.size
        traverse_node(root, total_size, len(root.children) == 0)

        # Create total row
        total_usage_display = (
            f"{total_size / factor:.2f}"
            if memory_unit != MemoryUnit.B
            else f"{int(total_size / factor)}"
        )
        overall_avg = (total_size / total_count) / factor if total_count > 0 else 0
        overall_min_conv = (overall_min or 0) / factor
        overall_max_conv = overall_max / factor

        total_row = TableRow(
            key="Total Keys Scanned",
            count=str(total_count),
            size=total_usage_display,
            avg_size=f"{overall_avg:.4f}" if memory_unit != MemoryUnit.B else f"{int(overall_avg)}",
            min_size=f"{overall_min_conv:.4f}"
            if memory_unit != MemoryUnit.B
            else f"{int(overall_min_conv)}",
            max_size=f"{overall_max_conv:.4f}"
            if memory_unit != MemoryUnit.B
            else f"{int(overall_max_conv)}",
            percentage="100.00",
            level=0,
        )

        return rows, total_row

    def generate_table(
        self, title: str, rows: list[TableRow], total_row: TableRow, memory_unit: MemoryUnit
    ) -> Table:
        """
        Generate a renderable table object with the given rows and total row.

        Args:
            title: Title for the table
            rows: List of data rows
            total_row: Summary row
            memory_unit: Memory unit for column headers

        Returns:
            Configured Rich Table object
        """
        table: Table = Table(title=title, box=box.MINIMAL)

        # Header row
        table.add_column("Key", justify="left")
        table.add_column("Count", justify="right", style="green")
        table.add_column(f"Size ({memory_unit.upper()})", justify="right", style="magenta")
        table.add_column(f"Avg Size ({memory_unit.upper()})", justify="right", style="orange1")
        table.add_column(f"Min Size ({memory_unit.upper()})", justify="right", style="yellow")
        table.add_column(f"Max Size ({memory_unit.upper()})", justify="right", style="red")
        table.add_column("Memory Usage (%)", justify="right", style="cyan")

        # Data rows
        for row in rows:
            # Apply style based on indentation level
            style = None
            if row.is_truncation_row:
                style = "dim"

            table.add_row(
                row.key,
                row.count,
                row.size,
                row.avg_size,
                row.min_size,
                row.max_size,
                row.percentage,
                style=style,
            )

        # Summary row
        table.add_section()
        table.add_row(
            total_row.key,
            total_row.count,
            total_row.size,
            total_row.avg_size,
            total_row.min_size,
            total_row.max_size,
            total_row.percentage,
            style="bold",
        )

        return table
