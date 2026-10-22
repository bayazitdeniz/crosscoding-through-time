"""Download and verify selected paper release assets, restoring workspace paths."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_BASE_URL = "https://github.com/bayazitdeniz/crosscoding-through-time/releases/download/crosscoding-artifacts-v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, resume=False):
    command = ["wget", "--quiet", "--show-progress"]
    if resume:
        command.append("--continue")
    subprocess.run([*command, "--output-document", str(path), url], check=True)


def install_archive(archive, destination, asset, manifest):
    """Verify all contents before extraction; keep existing matching files."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        expected = asset["files"]
        if len(zf.namelist()) != len(expected) or set(zf.namelist()) != set(expected):
            raise ValueError(f"Unexpected archive contents: {archive.name}")
        to_extract = []
        for name in expected:
            relative = Path(name)
            target = (destination / relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(destination):
                raise ValueError(f"Invalid archive path: {name}")
            metadata = manifest["files"][name]
            if zf.getinfo(name).file_size != metadata["size_bytes"]:
                raise ValueError(f"Wrong size for {name}")
            digest = hashlib.sha256()
            with zf.open(name) as f:
                for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                    digest.update(chunk)
            if digest.hexdigest() != metadata["sha256"]:
                raise ValueError(f"Checksum mismatch for {name}")
            if target.exists():
                if not target.is_file() or sha256(target) != metadata["sha256"]:
                    raise FileExistsError(f"Refusing to overwrite a different existing file: {target}")
            else:
                to_extract.append((name, target))
        for name, target in to_extract:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, CHUNK_SIZE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", "pythia", "olmo", "bloom"], default="all")
    parser.add_argument("--include-paper-results", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    parser.add_argument("--asset", action="append", help="Download a specific ZIP asset; may be repeated")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Override the release URL, e.g. to test a local mirror")
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--download-dir", type=Path)
    args = parser.parse_args()
    if not shutil.which("wget"):
        parser.error("wget is required")
    destination = args.destination.resolve()
    cache = args.download_dir or destination / "downloads/crosscoding-artifacts-v1"
    cache.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    manifest_path = cache / "release-manifest.json"
    download(f"{base_url}/release-manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assets = [a for a in manifest["assets"] if
              (a["kind"] == "weights" and not args.results_only and args.model in ("all", a.get("model"))) or
              (a["kind"] == "paper_results" and (args.include_paper_results or args.results_only))]
    if args.asset:
        assets = [a for a in manifest["assets"] if a["name"] in args.asset]
        if {a["name"] for a in assets} != set(args.asset):
            parser.error("An --asset name is missing from the release manifest")
    if not assets:
        parser.error("No matching assets in the release manifest")
    for asset in assets:
        name = asset["name"]
        if Path(name).name != name:
            raise ValueError(f"Invalid asset name: {name}")
        archive = cache / name
        if not archive.exists():
            partial = cache / f"{name}.partial"
            download(f"{base_url}/{name}", partial, resume=True)
            if partial.stat().st_size != asset["size_bytes"] or sha256(partial) != asset["sha256"]:
                raise ValueError(f"Download checksum or size mismatch: {partial}")
            partial.rename(archive)
        elif archive.stat().st_size != asset["size_bytes"] or sha256(archive) != asset["sha256"]:
            raise ValueError(f"Cached asset checksum or size mismatch: {archive}")
        install_archive(archive, destination, asset, manifest)
        print(f"Installed {name} into {destination}", flush=True)


if __name__ == "__main__":
    main()
