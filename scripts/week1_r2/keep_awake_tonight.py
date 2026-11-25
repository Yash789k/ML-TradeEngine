"""Temporary sleep assertion with an absolute expiry, including after relogin."""
import argparse
from datetime import datetime, timezone
import math
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--until', required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.until)
    if deadline.tzinfo is None:
        parser.error('Expiry must include a time zone')
    seconds = math.ceil((deadline - datetime.now(timezone.utc)).total_seconds())
    if seconds > 0:
        os.execv('/usr/bin/caffeinate', ['caffeinate', '-i', '-s', '-t', str(seconds)])


if __name__ == '__main__':
    main()
