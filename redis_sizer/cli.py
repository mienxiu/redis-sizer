import time
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

import typer
from redis import Redis
from redis.exceptions import RedisError
from rich import box
from rich.console import Console
from rich.progress import track
from rich.table import Table

app = typer.Typer()


class MemoryUnit(str, Enum):
    """
    For displaying memory usage in different units.
    """

    B = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"

    def get_factor(self) -> int:
        """
        Get the conversion factor for the memory unit.
        """
        if self == MemoryUnit.B:
            return 1
        elif self == MemoryUnit.KB:
            return 1024
        elif self == MemoryUnit.MB:
            return 1024 * 1024
        elif self == MemoryUnit.GB:
            return 1024 * 1024 * 1024
        else:
            raise ValueError("Invalid value for memory_unit. Use B, KB, MB, or GB.")


@dataclass
class TableRow:
    """
    Represent a row in the result table.
    """

    key: str
    count: str
    size: str
    avg_size: str
    min_size: str
    max_size: str
    percentage: str
    level: int = 0  # Add level for indentation
    is_truncation_row: bool = False  # dimmed if True


@dataclass
class KeyNode:
    """
    Represent a node in the key hierarchy tree.
    """

    name: str  # The display name for this node
    full_path: str  # The full path from root to this node
    level: int  # The depth level in the hierarchy (0 for root, 1 for first level, etc.)
    keys: list[str]  # List of actual Redis keys that belong directly to this node
    size: int  # Total memory size of all keys in this subtree (including children)
    sizes: list[int]  # List of individual memory sizes for statistics (e.g., min, max, avg)
    children: dict[str, "KeyNode"]  # Child nodes in the hierarchy, keyed by their name


@app.command()
def analyze(
    host: str,
    port: Annotated[int, typer.Option(help="Port number")] = 6379,
    db: Annotated[int, typer.Option(help="DB number")] = 0,
    password: Annotated[str | None, typer.Option(help="Password")] = None,
    socket_timeout: Annotated[int, typer.Option(help="Socket timeout in seconds")] = 10,
    socket_connect_timeout: Annotated[
        int, typer.Option(help="Socket connect timeout in seconds")
    ] = 10,
    pattern: Annotated[str, typer.Option(help="Pattern to filter keys")] = "*",
    sample_size: Annotated[int | None, typer.Option(help="Number of keys to sample")] = None,
    namespace_separator: Annotated[str, typer.Option(help="Separator for key namespaces")] = ":",
    memory_unit: Annotated[
        MemoryUnit, typer.Option(help="Memory unit for display in result table")
    ] = MemoryUnit.B,
    max_leaves: Annotated[
        int | None, typer.Option(help="Maximum number of leaf keys to display per namespace")
    ] = 5,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for scanning and calculating memory usage")
    ] = 1000,
):
    """
    Analyze memory usage across keys in a Redis database and display the results in a table.
    """
    # Start the timer to measure execution time
    start_time = time.time()

    # Create Console instance for rich output
    console = Console()

    # Create Redis client
    redis = Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_connect_timeout,
    )

    try:
        # Get the total number of keys in the database
        total_size: int = redis.dbsize()  # type: ignore
        if total_size == 0:
            console.print("[yellow]No keys found in the database.[/yellow]")
            redis.close()
            return
        console.print(f"The total number of keys: {total_size}")
        memory_usage = _get_memory_usage(
            redis=redis,
            pattern=pattern,
            batch_size=batch_size,
            sample_size=sample_size,
            console=console,
            total=total_size,
        )
    except RedisError as error:
        console.print(f"[red]Error occured: {error}[/red]")
        redis.close()
        exit(1)
    redis.close()

    if not memory_usage:
        console.print(f"[yellow]No keys found matching the pattern: {pattern}[/yellow]")
        return

    root = _build_key_tree(memory_usage, namespace_separator)
    rows, total_row = _generate_rows(root, memory_usage, memory_unit, max_leaves)
    table = _generate_table("Memory Usage", rows=rows, total_row=total_row, memory_unit=memory_unit)
    console.print(table)

    console.print(f"Took {(time.time() - start_time):.2f} seconds")


def _get_memory_usage(
    redis: Redis,
    pattern: str,
    batch_size: int,
    sample_size: int | None,
    console: Console,
    total: int | None = None,
) -> dict[str, int | None]:
    """
    Scan keys and get their memory usage using Lua script.
    Returns a dictionary mapping keys to their memory usage.
    """
    # Lua script that combines SCAN and MEMORY USAGE
    script = """
    local cursor = ARGV[1]
    local pattern = ARGV[2]
    local batch_size = tonumber(ARGV[3])
    
    -- Perform SCAN
    local scan_result = redis.call('SCAN', cursor, 'MATCH', pattern, 'COUNT', batch_size)
    local new_cursor = scan_result[1]
    local keys = scan_result[2]
    
    -- Get memory usage for each key
    local memory_usage = {}
    for i, key in ipairs(keys) do
        -- SAMPLES 0 means sampling the all of the nested values
        memory_usage[i] = redis.call('MEMORY', 'USAGE', key, 'SAMPLES', 0)
    end
    
    return {new_cursor, keys, memory_usage}
    """
    get_keys_and_memory = redis.register_script(script)

    # Progress tracking
    progress = track(
        range(sample_size or total or 0),
        description="Scanning and measuring keys...",
        console=console,
        show_speed=False,
    )
    progress_iter = iter(progress)
    next(progress_iter)  # Start the progress iterator

    memory_usage = {}
    cursor = 0
    collected_count = 0
    while True:
        # Call Lua script
        result: list = get_keys_and_memory(args=[cursor, pattern, batch_size])  # type: ignore
        new_cursor = int(result[0])
        keys = [k.decode() if isinstance(k, bytes) else k for k in result[1]]
        memory_values = result[2]

        # Sanity check
        assert len(keys) == len(memory_values), "Keys and memory values length mismatch"

        # Process results
        for key, memory_value in zip(keys, memory_values):
            if memory_value is not None:
                memory_usage[key] = memory_value
                collected_count += 1

                # Update progress
                try:
                    next(progress_iter)
                except StopIteration:
                    pass

        # Check if we've collected enough samples
        if sample_size and collected_count >= sample_size:
            # Trim to exact sample size if needed
            all_keys = list(memory_usage.keys())[:sample_size]
            memory_usage = {k: memory_usage[k] for k in all_keys}
            break

        # Check if scan is complete
        if new_cursor == 0:
            break

        cursor = new_cursor

    progress.close()  # type: ignore

    return memory_usage


def _build_key_tree(memory_usage: dict[str, int | None], separator: str) -> KeyNode:
    """
    Build a hierarchical tree structure from the memory usage dictionary.
    """
    root = KeyNode(name="", full_path="", level=0, keys=[], size=0, sizes=[], children={})

    for key in memory_usage.keys():
        size = memory_usage.get(key)
        if size is None:
            continue

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
    _propagate_sizes(root)

    return root


def _propagate_sizes(node: KeyNode) -> tuple[int, list[int]]:
    """
    Recursively propagate sizes from children to parents.
    Returns total size and list of all sizes for this subtree.
    """
    total_size = node.size
    all_sizes = node.sizes.copy()

    for child in node.children.values():
        child_size, child_sizes = _propagate_sizes(child)
        total_size += child_size
        all_sizes.extend(child_sizes)

    node.size = total_size
    node.sizes = all_sizes
    return total_size, all_sizes


def _generate_rows(
    root: KeyNode,
    memory_usage: dict[str, int | None],
    memory_unit: MemoryUnit,
    max_leaves: int | None,
) -> tuple[list[TableRow], TableRow]:
    """
    Generate table rows from the tree structure with proper indentation.
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
            key_sizes = [(k, memory_usage.get(k, 0)) for k in node.keys]
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
            sorted_children = sorted(node.children.items(), key=lambda x: x[1].size, reverse=True)

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


def _generate_table(
    title: str, rows: list[TableRow], total_row: TableRow, memory_unit: MemoryUnit
) -> Table:
    """
    Generate a renderable table object with the given rows and total row.
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


if __name__ == "__main__":
    app()
