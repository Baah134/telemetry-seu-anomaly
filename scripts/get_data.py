"""Download the official telemanom data.zip (Internet Archive snapshot of the original S3
URL, which now returns 403), verify its SHA-256 against the checksum published by
TimeSeAD, fetch labeled_anomalies.csv from khundman/telemanom, and unpack train/ and
test/ into data/raw/. Usage: python -m scripts.get_data
"""
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = ("http://web.archive.org/web/20221016205142id_/"
           "http://s3-us-west-2.amazonaws.com/telemanom/data.zip")
ZIP_SHA256 = "b4d66deb492d9b0a353b51879152687ed9313897e8e19320d2dc853d738ed8a7"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
RAW = Path("data/raw")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    z = RAW / "data.zip"
    if not z.exists():
        print("downloading", ZIP_URL)
        urllib.request.urlretrieve(ZIP_URL, z)
    got = sha256(z)
    if got != ZIP_SHA256:
        raise SystemExit(f"checksum mismatch: {got} != {ZIP_SHA256}; refusing to use the file")
    print("data.zip SHA-256 OK")
    if not (RAW / "labeled_anomalies.csv").exists():
        urllib.request.urlretrieve(LABELS_URL, RAW / "labeled_anomalies.csv")
    if not (RAW / "train").exists():
        tmp = RAW / "_unz"
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)
        for d in ("train", "test"):
            shutil.move(str(tmp / "data" / d), str(RAW / d))
        shutil.rmtree(tmp)
    print("data ready:", len(list((RAW / "train").glob("*.npy"))), "channels")


if __name__ == "__main__":
    main()
