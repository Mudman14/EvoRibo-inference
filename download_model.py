"""Download the versioned inference artifact and verify its SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('models/evoribo.onnx'))
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('model.json').read_text())
    if args.out.exists():
        parser.error('File already exists; choose another output path')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    partial = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.out.parent, suffix='.partial', delete=False) as output:
            partial = Path(output.name)
            with urllib.request.urlopen(manifest['url'], timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != manifest['sha256']:
            raise ValueError('Download checksum mismatch')
        # Do not overwrite a destination another process created meanwhile.
        with partial.open('rb') as source, args.out.open('xb') as target:
            import shutil
            shutil.copyfileobj(source, target)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    print(args.out)


if __name__ == '__main__':
    main()
