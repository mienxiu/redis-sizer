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
    namespace_level: Annotated[
        int,
        typer.Option(
            min=0,
            help="Maximum number of namespace levels to aggregate keys by. 0 means no aggregation.",
        ),
    ] = 0,
    memory_unit: Annotated[
        MemoryUnit, typer.Option(help="Memory unit for display in result table")
    ] = MemoryUnit.B,
    top: Annotated[
        int | None, typer.Option(help="Maximum number of rows to display in result table")
    ] = 10,
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

    for level in range(0, namespace_level + 1):
        # Aggregate memory usage for this namespace level
        group_data: dict[str, dict] = {}  # group -> {keys: [], size: 0, sizes: []}
        overall_min: int | None = None
        overall_max: int = 0
        valid_keys_count: int = 0
        for key in keys:
            memory_usage: int | None = memory_usage_by_key[key]
            if memory_usage is None:
                continue
            group: str = _parse_key_group(key, namespace_separator, level)
            if group not in group_data:
                group_data[group] = {"keys": [], "size": 0, "sizes": []}
            group_data[group]["keys"].append(key)
            group_data[group]["size"] += memory_usage
            group_data[group]["sizes"].append(memory_usage)
            overall_min = memory_usage if overall_min is None else min(overall_min, memory_usage)
            overall_max = max(overall_max, memory_usage)
            valid_keys_count += 1

        if valid_keys_count == 0:
            console.print(
                f"[yellow]No valid keys found for namespace level {level}. The scanned keys might have expired.[/yellow]"
            )
            continue

        # Sort the groups by size in descending order and limit if 'top' is specified
        sorted_groups = sorted(
            (
                (group, data["size"], data["sizes"], len(data["keys"]))
                for group, data in group_data.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        total_key_groups: int = len(sorted_groups)
        if top is not None and top > 0 and top < total_key_groups:
            group_data_for_rows = sorted_groups[:top]
            hidden_count = total_key_groups - top
        else:
            group_data_for_rows = sorted_groups
            hidden_count = 0

        # Prepare table rows for current namespace level
        rows: list[TableRow] = []
        factor: int = _get_memory_unit_factor(memory_unit)
        total_size: int = sum(group["size"] for group in group_data.values())
        for group, size, sizes, count in group_data_for_rows:
            display_key: str = group if level == 0 else f"{group}[dim]*[/dim]"
            size_converted: float = size / factor
            display_size: str = (
                f"{size_converted:.2f}" if memory_unit != MemoryUnit.B else f"{int(size_converted)}"
            )
            avg_usage = (sum(sizes) / len(sizes)) / factor if sizes else 0
            min_usage = min(sizes) / factor if sizes else 0
            max_usage = max(sizes) / factor if sizes else 0
            percentage = (size / total_size * 100) if total_size else 0
            rows.append(
                TableRow(
                    key=display_key,
                    count=str(count),
                    size=display_size,
                    avg_size=f"{avg_usage:.4f}"
                    if memory_unit != MemoryUnit.B
                    else f"{int(avg_usage)}",
                    min_size=f"{min_usage:.4f}"
                    if memory_unit != MemoryUnit.B
                    else f"{int(min_usage)}",
                    max_size=f"{max_usage:.4f}"
                    if memory_unit != MemoryUnit.B
                    else f"{int(max_usage)}",
                    percentage=f"{percentage:.2f}",
                )
            )

        # Prepare the total row
        total_usage_display: str = (
            f"{total_size / factor:.2f}"
            if memory_unit != MemoryUnit.B
            else f"{int(total_size / factor)}"
        )
        overall_avg: float = (total_size / valid_keys_count) / factor
        overall_min_conv: float = (overall_min or 0) / factor
        overall_max_conv: float = overall_max / factor
        overall_avg_display: str = (
            f"{overall_avg:.4f}" if memory_unit != MemoryUnit.B else f"{int(overall_avg)}"
        )
        overall_min_display: str = (
            f"{overall_min_conv:.4f}" if memory_unit != MemoryUnit.B else f"{int(overall_min_conv)}"
        )
        overall_max_display: str = (
            f"{overall_max_conv:.4f}" if memory_unit != MemoryUnit.B else f"{int(overall_max_conv)}"
        )
        total_count: int = sum(len(group["keys"]) for group in group_data.values())
        total_row: TableRow = TableRow(
            key="Total Keys Scanned",
            count=str(total_count),
            size=total_usage_display,
            avg_size=overall_avg_display,
            min_size=overall_min_display,
            max_size=overall_max_display,
            percentage="100.00",
        )

        _print_memory_usage_table(
            title=f"Memory Usage Table for Namespace Level {level}",
            rows=rows,
            total_row=total_row,
            hidden_count=hidden_count,
            memory_unit=memory_unit,
            show_extra_stats=(level != 0),
            console=console,
        )
        console.print("")  # Add a blank line between tables

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


def _parse_key_group(key: str, separator: str, level: int) -> str:
    """
    Parse the key group of a key up to the specified level using the given separator.
    For example, _parse_key_group("a:b:c:d", ":", 2) -> "a:b"

    If level is 0, return the entire key.
    """
    if level == 0:
        return key
    return separator.join(key.split(separator)[:level])


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


def _print_memory_usage_table(
    title: str,
    rows: list[TableRow],
    total_row: TableRow,
    hidden_count: int,
    memory_unit: MemoryUnit,
    show_extra_stats: bool,  # new parameter to control detail columns
    console: Console,
):
    """
    - show_extra_stats: If True, show (Avg, Min, Max) size in the table.
    """
    table: Table = Table(title=title, box=box.MINIMAL)
    table.add_column("Key", justify="left")
    table.add_column("Count", justify="right", style="green")
    table.add_column(f"Size ({memory_unit.upper()})", justify="right", style="magenta")
    if show_extra_stats:
        table.add_column(f"Avg Size ({memory_unit.upper()})", justify="right", style="orange1")
        table.add_column(f"Min Size ({memory_unit.upper()})", justify="right", style="yellow")
        table.add_column(f"Max Size ({memory_unit.upper()})", justify="right", style="red")
    table.add_column("Memory Usage (%)", justify="right", style="cyan")

    for row in rows:
        if show_extra_stats:
            table.add_row(
                row.key,
                row.count,
                row.size,
                row.avg_size,
                row.min_size,
                row.max_size,
                row.percentage,
            )
        else:
            table.add_row(row.key, row.count, row.size, row.percentage)

    if hidden_count > 0:
        if show_extra_stats:
            table.add_row(
                f"... {hidden_count} more entries not shown ...",
                "",
                "",
                "",
                "",
                "",
                "",
                style="dim",
            )
        else:
            table.add_row(f"... {hidden_count} more entries not shown ...", "", "", "", style="dim")
    table.add_section()

    if show_extra_stats:
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
    else:
        table.add_row(
            total_row.key,
            total_row.count,
            total_row.size,
            total_row.percentage,
            style="bold",
        )
    console.print(table)


if __name__ == "__main__":
    app()
