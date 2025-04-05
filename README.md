# redis-sizer

A simple command-line tool for analyzing memory usage across keys in a Redis database.

![Sample output](https://raw.githubusercontent.com/mienxiu/redis-sizer/refs/heads/main/docs/sample_output.png)

redis-sizer can help with the following problems:
- Optimize caching strategy.
- Gain insights into memory usage at different namespace levels.
- Identify which parts of your application consume the most memory.

## Installation

Install via pip:

```bash
pip install redis-sizer
```

## Usage

Run redis-sizer by specifying the Redis host along with any desired options:

```bash
redis-sizer [OPTIONS] HOST
```

Options:

- `--port`: Port number [default: 6379]
- `--db`: DB number [default: 0]
- `--password`: Password [default: None]
- `--socket-timeout`: Socket timeout in seconds [default: 10]
- `--socket-connect-timeout`: Socket connect timeout in seconds [default: 10]
- `--pattern`: Pattern to filter keys [default: *]
- `--sample-size`: Number of keys to sample [default: None]
- `--namespace-separator`: Separator for key namespaces [default: :]
- `--namespace-level`: Maximum number of namespace levels to aggregate keys by. 0 means no aggregation. [default: 0]
- `--memory-unit`: Memory unit for display in result table [default: B]
- `--top`: Maximum number of rows to display in result table [default: 10]
- `--scan-count`: `COUNT` option for scanning keys [default: 1000]
- `--batch-size`: Batch size for calculating memory usage [default: 1000]

### Aggregating Memory Usage by Key Groups

Many applications using Redis adopt hierarchical key naming conventions.
redis-sizer leverages this structure to aggregate and analyze memory usage, helping you quickly identify which key groups consume the most memory.

For example, if your keys follow a colon-delimited format (e.g., `users:profiles:...`, `users:logs:...`, `sessions:active:...`), the hierarchy can be visualized like a tree structure:

```
DB
├── users:
│   ├── profiles:
│   │   ├── ...
│   └── logs:
│       ├── ...
├── sessions:
│   └── active:
...
```

Using `--namespace-level 2` prints out memory usage tabels up to the second-level namespace:

![Example output](https://raw.githubusercontent.com/mienxiu/redis-sizer/refs/heads/main/docs/example_output.png)

## Considerations

- Since data can change during analysis, the final result may not accurately reflect the database’s state at the beginning of the process.
- For databases with large collections of keys, consider using the `--sample-size` option to balance processing time with accuracy. If `--sample-size` is not specified, redis-sizer will perform perform a full iteration.
- Increasing `--scan-count` or `--batch-size` can reduce network overhead. However, this can increase the load on the Redis server without providing meaningful performance improvements and, in some cases, may even lead to server blocking. Adjust these values carefully based on your server’s capacity.

## Redis Version Compatibility

- This tool relies on the Redis [`MEMORY USAGE`](https://redis.io/docs/latest/commands/memory-usage/) command, which has been available since version 4.0.0.
- It has been tested and verified to work with Redis versions 4, 5, 6, and 7.

## Dependencies

redis-sizer depends on:

- [redis-py](https://github.com/redis/redis-py): to communicate with the Redis database.
- [typer](https://github.com/fastapi/typer): to build CLI application, with output formatting supported by [rich](https://github.com/Textualize/rich)
