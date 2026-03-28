import argparse
from pathlib import Path
import mimetypes    # media type

from lib.llm import LlmPrompt

img_path = Path(__file__).resolve().parents[1]
def main() -> None:   
    parser = argparse.ArgumentParser(description="Describe Image")
    parser.add_argument("--image", type=str, help="path to an image file")
    parser.add_argument("--query", type=str, help="a text query to rewrite based on the image")


    args = parser.parse_args()

    #  a media type, content type or MIME type is a two-part identifier for file formats and content formats. 
    # Their purpose is comparable to filename extensions and uniform type identifiers, in that they identify the intended data format. 
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(img_path/args.image, 'rb')as f:
        image_bytes = f.read()  # read an image in bytes
    
    response = LlmPrompt('gemma-3-27b-it').image_search(args.query, image_bytes, mime)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == '__main__':
    main()