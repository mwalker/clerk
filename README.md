# Qwen

Process images using Qwen2-VL-7B model with optional resizing and custom prompt.

## Usage

```
usage: clerk.py [-h] [--max-size MAX_SIZE] [--prompt PROMPT] [--debug] [--space SPACE] [--duplicate-space] [--model MODEL] images [images ...]

Process images using a specified Hugging Face space and model with optional resizing, rotation, and custom prompt.

positional arguments:
  images               Paths to image files to process

options:
  -h, --help           show this help message and exit
  --max-size MAX_SIZE  Maximum size for image dimension (default: 1280)
  --prompt PROMPT      Prompt for the model (default: 'Extract text')
  --debug              Enable debug mode
  --space SPACE        Hugging Face space to use (default: 'GanymedeNil/Qwen2-VL-7B')
  --duplicate-space    Use Client.duplicate() to create the client
  --model MODEL        Model to use for prediction (default: 'Qwen/Qwen2-VL-7B-Instruct')

Environment Variables:
  HF_TOKEN: If set, this Hugging Face API token will be used for authentication.
            You can set it by running 'export HF_TOKEN=your_token' before running this script.
```
