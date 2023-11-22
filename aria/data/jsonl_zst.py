import builtins
import contextlib
import io
import zstandard
import jsonlines
import json


class Reader:
    """Reader for the jsonl.zst format."""

    def __init__(self, path: str):
        """Initializes the reader.

        Args:
            path (str): Path to the file.
        """
        self.path = path

    def __iter__(self):
        with builtins.open(self.path, "rb") as fh:
            cctx = zstandard.ZstdDecompressor()
            reader = io.BufferedReader(cctx.stream_reader(fh))
            yield from jsonlines.Reader(reader)


class Writer:
    """Writer for the jsonl.zst format."""

    def __init__(self, path: str):
        """Initializes the writer.

        Args:
            path (str): Path to the file.
        """
        self.path = path

    def __enter__(self):
        self.fh = builtins.open(self.path, "wb")
        self.cctx = zstandard.ZstdCompressor()
        self.compressor = self.cctx.stream_writer(self.fh)
        return self

    def write(self, obj):
        self.compressor.write(json.dumps(obj).encode("UTF-8") + b"\n")

    def __exit__(self, exc_type, exc_value, traceback):
        self.compressor.flush(zstandard.FLUSH_FRAME)
        self.fh.flush()
        self.compressor.close()
        self.fh.close()


@contextlib.contextmanager
def open(path: str, mode: str = "r"):
    """Read/Write a jsonl.zst file.

    Args:
        path (str): Path to the file.
        mode (str): Mode to open the file in. Only 'r' and 'w' are supported.

    Returns:
        Reader or Writer: Reader if mode is 'r', Writer if mode is 'w'.
    """
    if mode == "r":
        yield Reader(path)
    elif mode == "w":
        with Writer(path) as writer:
            yield writer
    else:
        raise ValueError(f"Unsupported mode '{mode}'")
