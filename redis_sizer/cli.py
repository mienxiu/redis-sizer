import time
from typing import Annotated

import typer
from redis import Redis
from redis.exceptions import RedisError
from rich.console import Console
from rich.progress import Progress

from redis_sizer.models import MemoryUnit
from redis_sizer.table_formatter import TableFormatter
from redis_sizer.tree_builder import KeyTreeBuilder

app = typer.Typer()


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
        decode_responses=True,
    )

    try:
        # Get the total number of keys in the database
        total_size: int = redis.dbsize()  # type: ignore
        if total_size == 0:
            console.print("[yellow]No keys found in the database.[/yellow]")
            return
        console.print(f"The total number of keys: {total_size}")
        memory_usage = _get_memory_usage(
            redis=redis,
            pattern=pattern,
            batch_size=batch_size,
            sample_size=sample_size,
            total=total_size,
            console=console,
        )
        if not memory_usage:
            console.print(f"[yellow]No keys found matching the pattern: {pattern}[/yellow]")
            return
    except RedisError as error:
        console.print(f"[red]Error occured: {error}[/red]")
        exit(1)
    finally:
        redis.close()

    tree_builder = KeyTreeBuilder()
    root = tree_builder.build_tree(memory_usage, namespace_separator)

    table_formatter = TableFormatter()
    rows, total_row = table_formatter.generate_rows(root, memory_usage, memory_unit, max_leaves)
    table = table_formatter.generate_table("Memory Usage", rows, total_row, memory_unit)
    console.print(table)

    console.print(f"Took {(time.time() - start_time):.2f} seconds")


def _get_memory_usage(
    redis: Redis,
    pattern: str,
    batch_size: int,
    sample_size: int | None,
    total: int,
    console: Console,
) -> dict[str, int]:
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

    cursor = 0
    memory_usage = {}
    collected_count = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Scanning and measuring keys...", total=sample_size or total)

        while True:
            # Call Lua script
            result: list = get_keys_and_memory(args=[cursor, pattern, batch_size])  # type: ignore
            new_cursor = int(result[0])
            keys = result[1]
            memory_values = result[2]

            # Sanity check
            assert len(keys) == len(memory_values), "Keys and memory values length mismatch"

            # Update memory_usage dictionary
            batch_processed = 0
            for key, memory_value in zip(keys, memory_values):
                if memory_value is not None:
                    memory_usage[key] = memory_value
                    collected_count += 1
                    batch_processed += 1

            # Update progress bar by the number of keys processed in this batch
            if batch_processed > 0:
                progress.update(task, advance=batch_processed)

            # Check if we've collected enough samples
            if sample_size and collected_count >= sample_size:
                # Trim to exact sample size
                all_keys = list(memory_usage.keys())[:sample_size]
                memory_usage = {k: memory_usage[k] for k in all_keys}
                break

            # Check if scan is complete
            if new_cursor == 0:
                break

            cursor = new_cursor

    return memory_usage


if __name__ == "__main__":
    app()
