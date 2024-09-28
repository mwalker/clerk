# clerk

Transcribe text from images using a vision LLM, currently Qwen/Qwen2-VL-7B-Instruct.

Will resize the images if needed, and if they have xattr genereted by
Spin it will also perform a rotation. 

Your prompt must try to get the model to return JSON, and clerk will validate
the JSON that is returned. Optionally clerk can also validate the JSON
against
a JSON Schema.

## Prerequisites

Assumes uv is installed. See https://github.com/astral-sh/uv

## Usage

```
usage: uv run clerk.py [-h] [--max-size MAX_SIZE] [--prompt PROMPT] [--debug] [--space SPACE] [--duplicate-space] [--model MODEL] [--schema SCHEMA] [--repeat] images [images ...]

Process images using a specified Hugging Face space and model with optional resizing, rotation, custom prompt, and JSON schema validation.

positional arguments:
  images               Paths to image files to process

options:
  -h, --help           show this help message and exit
  --max-size MAX_SIZE  Maximum size for image dimension (default: 1280)
  --prompt PROMPT      Prompt for the model (default: 'Extract text')
  --debug              Enable debug mode
  --space SPACE        Hugging Face space to use (default: 'GanymedeNil/Qwen2-VL-7B')
  --duplicate-space    Use Client.duplicate() to create the client in your own space
  --model MODEL        Model to use for prediction (default: 'Qwen/Qwen2-VL-7B-Instruct')
  --schema SCHEMA      Path or URL to JSON schema for validation
  --repeat             Process images even if they have already been processed

Environment Variables:
  HF_TOKEN: If set, this Hugging Face API token will be used for authentication.
            You can set it by running 'export HF_TOKEN=your_token' before running this script.
```

## License

clerk is licensed under the Apache License, Version 2.0, ([LICENSE](LICENSE) or
https://www.apache.org/licenses/LICENSE-2.0).
