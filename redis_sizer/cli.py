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


@dataclass
class KeyNode:
    """
    Represent a node in the key hierarchy tree.
    """

    name: str
    full_path: str
    level: int
    keys: list[str]
    size: int
    sizes: list[int]
    children: dict[str, "KeyNode"]


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
    top: Annotated[
        int | None, typer.Option(help="Maximum number of rows to display in result table")
    ] = 5,
    scan_count: Annotated[int, typer.Option(help="COUNT option for scanning keys")] = 1000,
    batch_size: Annotated[int, typer.Option(help="Batch size for calculating memory usage")] = 1000,
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

    # Get the total number of keys in the database
    try:
        total_size: int = redis.dbsize()  # type: ignore
    except RedisError as error:
        console.print(f"[red]Error occured: {error}[/red]")
        redis.close()
        exit(1)

    # If the total size is 0, stop the process
    if total_size == 0:
        console.print("[yellow]No keys found in the database.[/yellow]")
        redis.close()
        return

    console.print(f"The total number of keys: {total_size}")

    # Scan the keys in the database
    keys = _scan_keys(
        redis=redis,
        pattern=pattern,
        count=scan_count,
        sample_size=sample_size,
        console=console,
        total=total_size,
    )
    if not keys:
        console.print(f"[yellow]No keys found matching the pattern: {pattern}[/yellow]")
        redis.close()
        return

    # Get memory usage for each key (in batches)
    memory_usage_by_key: dict[str, int | None] = {}  # key -> memory usage
    for i in track(
        range(0, len(keys), batch_size),
        description="Calculating memory usage...",
        console=console,
    ):
        batch = keys[i : i + batch_size]
        try:
            memory_usage_by_key.update(_get_memory_usage(redis=redis, keys=batch))
        except RedisError as error:
            console.print(f"[red]Error occured: {error}[/red]")
            redis.close()
            exit(2)

    redis.close()

    # Build hierarchical tree structure
    root = _build_key_tree(keys, memory_usage_by_key, namespace_separator)

    if not root.children and not root.keys:
        console.print("[yellow]No valid keys found. The scanned keys might have expired.[/yellow]")
        return

    # Generate table rows from tree
    rows, total_row = _generate_hierarchical_rows(root, memory_usage_by_key, memory_unit, top)

    # Print the hierarchical table
    _print_hierarchical_table(
        title="Memory Usage Hierarchical View",
        rows=rows,
        total_row=total_row,
        memory_unit=memory_unit,
        console=console,
    )

    console.print(f"Took {(time.time() - start_time):.2f} seconds")


def _scan_keys(
    redis: Redis,
    pattern: str,
    count: int,
    sample_size: int | None,
    console: Console,
    total: int | None = None,
) -> list[str]:
    """
    Scan keys in the Redis database using the given pattern.
    If sample_size is None, perform a full iteration and estimate progress using total.
    """
    keys = []
    for key in track(
        redis.scan_iter(match=pattern, count=count),
        description="Scanning keys...",
        total=sample_size or total,
        console=console,
        show_speed=False,
    ):
        keys.append(key.decode())
        if sample_size and sample_size <= len(keys):
            break
    return keys


def _get_memory_usage(redis: Redis, keys: list[str]) -> dict[str, int | None]:
    """
    Get memory usage for a list of keys using Lua script.
    If the key doesn't exist (expired), memory usage is None.
    """
    script = """
    local result = {}
    for i, key in ipairs(KEYS) do
        result[i] = redis.call('MEMORY', 'USAGE', key, 'SAMPLES', 0)
    end
    return result
    """
    func = redis.register_script(script)
    return dict(zip(keys, func(keys=keys)))  # type: ignore


def _build_key_tree(
    keys: list[str], memory_usage_by_key: dict[str, int | None], separator: str
) -> KeyNode:
    """
    Build a hierarchical tree structure from keys.
    """
    root = KeyNode(name="", full_path="", level=0, keys=[], size=0, sizes=[], children={})

    for key in keys:
        memory_usage = memory_usage_by_key.get(key)
        if memory_usage is None:
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
            current_node.size += memory_usage
            current_node.sizes.append(memory_usage)
        else:
            # Key has no namespace, add to root
            root.keys.append(key)
            root.size += memory_usage
            root.sizes.append(memory_usage)

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


def _generate_hierarchical_rows(
    root: KeyNode,
    memory_usage_by_key: dict[str, int | None],
    memory_unit: MemoryUnit,
    top: int | None,
) -> tuple[list[TableRow], TableRow]:
    """
    Generate table rows from the tree structure with proper indentation.
    """
    rows: list[TableRow] = []
    factor = _get_memory_unit_factor(memory_unit)

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
            key_sizes = [(k, memory_usage_by_key.get(k, 0)) for k in node.keys]
            key_sizes.sort(key=lambda x: 0 if x[1] is None else x[1], reverse=True)

            # Apply top limit only at leaf level
            if is_last_level and top and len(key_sizes) > top:
                displayed_keys = key_sizes[:top]
                hidden_count = len(key_sizes) - top
            else:
                displayed_keys = key_sizes
                hidden_count = 0

            # Add rows for direct keys
            for key, size in displayed_keys:
                if size == 0:
                    continue

                display_key = "  " * node.level + key
                size_converted = size or 0 / factor
                display_size = (
                    f"{size_converted:.2f}"
                    if memory_unit != MemoryUnit.B
                    else f"{int(size_converted)}"
                )
                percentage = (size or 0 / parent_size * 100) if parent_size else 0

                rows.append(
                    TableRow(
                        key=display_key,
                        count="1",
                        size=display_size,
                        avg_size=display_size,
                        min_size=display_size,
                        max_size=display_size,
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


def _get_memory_unit_factor(memory_unit: MemoryUnit) -> int:
    """
    Get the conversion factor for the specified memory unit.
    """
    if memory_unit == MemoryUnit.B:
        return 1
    elif memory_unit == MemoryUnit.KB:
        return 1024
    elif memory_unit == MemoryUnit.MB:
        return 1024 * 1024
    elif memory_unit == MemoryUnit.GB:
        return 1024 * 1024 * 1024
    else:
        raise ValueError("Invalid value for memory_unit. Use B, KB, MB, or GB.")


def _print_hierarchical_table(
    title: str,
    rows: list[TableRow],
    total_row: TableRow,
    memory_unit: MemoryUnit,
    console: Console,
):
    """
    Print hierarchical table with indented keys.
    """
    table: Table = Table(title=title, box=box.MINIMAL)
    table.add_column("Key", justify="left")
    table.add_column("Count", justify="right", style="green")
    table.add_column(f"Size ({memory_unit.upper()})", justify="right", style="magenta")
    table.add_column(f"Avg Size ({memory_unit.upper()})", justify="right", style="orange1")
    table.add_column(f"Min Size ({memory_unit.upper()})", justify="right", style="yellow")
    table.add_column(f"Max Size ({memory_unit.upper()})", justify="right", style="red")
    table.add_column("Memory Usage (%)", justify="right", style="cyan")

    for row in rows:
        # Apply style based on indentation level
        style = None
        if "..." in row.key:
            style = "dim"
        elif row.level == 1:
            style = "bold"

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
    console.print(table)


if __name__ == "__main__":
    app()
