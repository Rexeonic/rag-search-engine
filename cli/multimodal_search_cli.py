import argparse

from lib.multimodal_search import (MultimodalSearch,
                                   verify_image_embedding,
                                   image_search_command)



def main() -> None:

    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser("verify_image_embedding", help="Search for non-text queries i.e image")
    verify_image_parser.add_argument("image_path", type=str, help="path to an image file")

    image_search_parser = subparsers.add_parser("image_search", help="Search for non-text queries i.e image")
    image_search_parser.add_argument("image_path", type=str, help="path to an image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)

        case "image_search":
            result = image_search_command(args.image_path)
        
            for i, res in enumerate(result):
                 print(f"{i+1}. {res['title']} (similarity: {res['similarity_score']})")
                 print(f"{res['description']}")
                 
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()