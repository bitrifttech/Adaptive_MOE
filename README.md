# Adaptive Mixture of Experts (MoE) System

An adaptive mixture of experts system that can dynamically create and manage domain-specific expert models based on incoming queries.

## Features

- **Base Model Integration**: Built on top of large language models (e.g., Mistral 7B)
- **Dynamic Expert Creation**: Automatically creates domain-specific experts when needed
- **Efficient Routing**: Smart routing to the most appropriate expert or base model
- **Resource Management**: Efficient memory and compute resource utilization
- **Extensible Architecture**: Easy to add new expert types and routing strategies

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/adaptive-moe.git
   cd adaptive-moe
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Usage

### Configuration

Create a configuration file (e.g., `config.yaml`) with your settings:

```yaml
model:
  base_model_id: "mistralai/Mistral-7B-v0.1"
  load_in_8bit: true

router:
  confidence_threshold: 0.7

expert:
  lora_rank: 16
  max_experts_in_memory: 5

logging:
  level: "INFO"
  log_dir: "logs"
```

### Command Line Interface

#### Interactive Mode

```bash
python -m adaptive_moe interactive --config config.yaml
```

#### Start API Server

```bash
python -m adaptive_moe serve --config config.yaml --host 0.0.0.0 --port 8000
```

#### Train a New Expert

```bash
python -m adaptive_moe train --config config.yaml --data-dir data/domain_data --output-dir experts/domain_expert
```

## Project Structure

```
adaptive_moe/
├── __init__.py         # Package initialization
├── __main__.py         # CLI entry point
├── router/             # Router implementation
├── experts/            # Expert models
├── integration/        # Response integration
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── config.py      # Configuration management
│   ├── logging.py     # Logging setup
│   └── model_utils.py # Model utilities
└── tests/              # Test files
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
isort .
```

### Type Checking

```bash
mypy .
```

## License

MIT
