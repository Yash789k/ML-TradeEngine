"""Finite, restart-safe idle-sleep assertion for the entire registered pilot.

Display sleep remains allowed. A closed lid, shutdown or exhausted battery still
requires the user; this does not claim physical wake or remote hosting.
"""
import argparse
from datetime import datetime, timezone
import math
import os


def remaining_seconds(until, now=None):
    deadline = datetime.fromisoformat(until)
    if deadline.tzinfo is None:
        raise ValueError('Expiry must include a time zone')
    return max(0, math.ceil((deadline - (now or datetime.now(timezone.utc))).total_seconds()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--until', required=True)
    args = parser.parse_args()
    seconds = remaining_seconds(args.until)
    if seconds:
        os.execv('/usr/bin/caffeinate', ['caffeinate', '-i', '-s', '-t', str(seconds)])


if __name__ == '__main__':
    main()
