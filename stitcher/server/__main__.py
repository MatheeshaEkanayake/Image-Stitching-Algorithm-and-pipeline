"""``python -m stitcher.server [--host H] [--port P]`` (needs the ``server`` extra)."""

import argparse

import uvicorn

from stitcher.server.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the image-stitch server.")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to serve a LAN/phone.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
