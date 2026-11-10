import gzip
import re
import urllib.request
from pathlib import Path
from datetime import datetime
import pandas as pd

DATA = Path("data")
DATA.mkdir(exist_ok=True)

SOURCES = {
    "NASA_access_log_Jul95.gz": "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
    "NASA_access_log_Aug95.gz": "https://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz",
}
# NASA log format: host - - [01/Jul/1995:00:00:01 -0400] "GET /path HTTP/1.0" 200 6245
LOG_RE = re.compile(
    r'^\S+ \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<request>[^"]*)" (?P<status>\d{3}|-) (?P<bytes>\d+|-)'
)
TIME_FORMAT = "%d/%b/%Y:%H:%M:%S %z"

def download_if_missing(name, url):
    path = DATA / name
    if path.exists() and path.stat().st_size > 1000:
        print(f"Using existing {path}")
        return path
    print(f"Downloading {name} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        if path.exists():
            path.unlink()
        raise RuntimeError(
            f"Could not download {name}: {exc}\n"
            "Download it manually from https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html "
            f"and place it at {path.resolve()}"
        )
    return path

def parse_logs(paths):
    timestamps = []
    total = 0
    bad = 0
    for path in paths:
        print(f"Parsing {path} ...")
        with gzip.open(path, "rt", encoding="latin-1", errors="replace") as fh:
            for line in fh:
                total += 1
                match = LOG_RE.match(line)
                if not match:
                    bad += 1
                    continue
                try:
                    dt = datetime.strptime(match.group("time"), TIME_FORMAT)
                except ValueError:
                    bad += 1
                    continue
                timestamps.append(dt.replace(tzinfo=None))
    if not timestamps:
        raise RuntimeError("No valid NASA log records parsed. Check the downloaded files.")
    series = pd.Series(1, index=pd.DatetimeIndex(timestamps))
    per_minute = series.resample("1min").sum().fillna(0).astype(int)
    out = pd.DataFrame({
        "timestamp": per_minute.index,
        "request_count": per_minute.values,
    })
    out["request_rate_rps"] = out["request_count"] / 60.0
    out.to_csv(DATA / "nasa_workload_1min.csv", index=False)
    print(f"Parsed {len(timestamps):,} requests from {total:,} lines; skipped {bad:,}.")
    print(f"Saved {len(out):,} one-minute intervals to data/nasa_workload_1min.csv")
    print(out.head())
    return out

if __name__ == "__main__":
    paths = [download_if_missing(n, u) for n, u in SOURCES.items()]
    parse_logs(paths)
