# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio-client",
#     "pillow",
# ]         
# ///

import sys
import argparse
import os
import time
import json
import re
from pathlib import Path
from PIL import Image
from gradio_client import Client, handle_file
from datetime import datetime

def get_image_format(file_path):
    format_map = {
        '.jpg': 'JPEG',
        '.jpeg': 'JPEG',
        '.png': 'PNG',
        '.gif': 'GIF',
        '.bmp': 'BMP',
        '.tiff': 'TIFF',
    }
    ext = file_path.suffix.lower()
    return format_map.get(ext, 'JPEG')

def resize_and_save_image(image_path, max_size):
    with Image.open(image_path) as img:
        original_size = img.size
        ratio = max_size / max(original_size)
        if ratio < 1:  # Only resize if the image is larger than max_size
            new_size = tuple(int(dim * ratio) for dim in original_size)
            img = img.resize(new_size, Image.LANCZOS)
            print(f"Resized {image_path} from {original_size} to {new_size}")
            
            # Create the new filename with 'resized' suffix
            new_filename = image_path.stem + "_resized" + image_path.suffix
            new_path = image_path.parent / new_filename
            
            # Save the resized image
            img.save(new_path)
            print(f"Saved resized image as {new_path}")
            
            return new_path
        else:
            print(f"{image_path} does not need resizing")
            return image_path

def save_response_to_file(image_path, response):
    response_file = image_path.with_name(f"{image_path.stem}_qwen.txt")
    with open(response_file, 'w', encoding='utf-8') as f:
        f.write(str(response))
    print(f"Saved API response to {response_file}")
    return response_file

def validate_json_structure(json_data):
    required_keys = ["identifier", "date", "correspondent", "summary"]
    
    if not isinstance(json_data, dict):
        return False, "JSON must be a single object"
    
    for key in required_keys:
        if key not in json_data:
            return False, f"Missing required key: {key}"
        if not isinstance(json_data[key], str):
            return False, f"Value for '{key}' must be a string"
    
    # Validate ISO8601 date
    try:
        datetime.fromisoformat(json_data["date"])
    except ValueError:
        return False, "Invalid ISO8601 date format"
    
    # Check 'tags' if present
    if "tags" in json_data:
        if not isinstance(json_data["tags"], list):
            return False, "'tags' must be an array"
        if not all(isinstance(tag, str) for tag in json_data["tags"]):
            return False, "All items in 'tags' must be strings"
    
    return True, "JSON structure is valid"

def extract_and_validate_json(response_file, image_path):
    with open(response_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for JSON code block
    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            json_data = json.loads(json_str)
            is_valid, message = validate_json_structure(json_data)
            
            if is_valid:
                print("Extracted JSON data is valid.")
                
                # Save JSON to file
                json_file = image_path.with_suffix('.json')
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
                print(f"Saved JSON data to {json_file}")
                
                # Print JSON to console
                print("Validated JSON data:")
                print(json.dumps(json_data, indent=2))
                
                return json_data
            else:
                print(f"Error: {message}")
        except json.JSONDecodeError as e:
            print(f"Error: Extracted JSON data is not valid. {str(e)}")
    else:
        print("Error: No JSON code block found in the response.")
    return None

def process_image(image_path, client, max_size, prompt):
    try:
        # Resize the image if necessary and get the path of the image to process
        image_to_process = resize_and_save_image(image_path, max_size)

        # Time the API call
        start_time = time.time()
        
        # Make the API call, wrapping image_to_process with handle_file
        result = client.predict(
            image=handle_file(str(image_to_process)),
            text_input=prompt,
            model_id="Qwen/Qwen2-VL-7B-Instruct",
            api_name="/run_example"
        )
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

        print(f"Successfully processed {image_to_process}")
        print(f"API call took {elapsed_time:.3f} milliseconds")

        # Save the response to a file
        response_file = save_response_to_file(image_path, result)

        # Extract, validate, and save JSON from the response
        extract_and_validate_json(response_file, image_path)

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Process images using Qwen2-VL-7B model with optional resizing and custom prompt.",
        epilog="Environment Variables:\n"
               "  HF_TOKEN: If set, this Hugging Face API token will be used for authentication.\n"
               "            You can set it by running 'export HF_TOKEN=your_token' before running this script.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", nargs="+", help="Paths to image files to process")
    parser.add_argument("--max-size", type=int, default=1280, help="Maximum size for image dimension (default: 1280)")
    parser.add_argument("--prompt", type=str, default="Extract text", help="Prompt for the Qwen2-VL-7B model (default: 'Extract text')")
    
    args = parser.parse_args()

    # Check for HF_TOKEN in environment variables
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print("Using Hugging Face API token from environment variable.")
    else:
        print("No Hugging Face API token found in environment. Some features may be limited.")

    # Initialize the Gradio client with the HF token if available
    client = Client("GanymedeNil/Qwen2-VL-7B", hf_token=hf_token)

    for image_path in args.images:
        path = Path(image_path)
        if path.is_file():
            process_image(path, client, args.max_size, args.prompt)
        else:
            print(f"File not found: {image_path}")

if __name__ == "__main__":
    main()
