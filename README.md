# Qwen

Process images using Qwen2-VL-7B model with optional resizing and custom prompt.

## Usage

```
usage: qwen.py [-h] [--max-size MAX_SIZE] [--prompt PROMPT] images [images ...]

Process images using Qwen2-VL-7B model with optional resizing and custom prompt.

positional arguments:
  images               Paths to image files to process

options:
  -h, --help           show this help message and exit
  --max-size MAX_SIZE  Maximum size for image dimension (default: 1280)
  --prompt PROMPT      Prompt for the Qwen2-VL-7B model (default: 'Extract text')
```
