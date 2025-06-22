# 🏀 HoopGT SDK 

## Installation

```bash
# Install dependencies
pip install -e .

# Or with uv
uv sync
```

## Usage

```bash
# Show help
hoopgt --help

# Optimize a model (placeholder)
hoopgt optimize ./model.pt --level balanced

# Deploy a model (placeholder)  
hoopgt deploy my-model --port 3000

# List models (placeholder)
hoopgt list

# Show system info
hoopgt info
```

## Development

This is a minimal boilerplate. Add your implementation in:

- `hoopgt/cli.py` - Main CLI commands
- `hoopgt/` - Add new modules as needed

## Commands Structure

- `optimize` - Model optimization logic
- `deploy` - Model deployment logic  
- `list` - Model registry/listing
- `info` - System information

## TODO

- [ ] Add optimization implementation
- [ ] Add deployment implementation
- [ ] Add model registry
- [ ] Add hardware detection
- [ ] Add quantization logic

Start building your MVP! 🚀
