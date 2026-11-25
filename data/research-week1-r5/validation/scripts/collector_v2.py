#!/usr/bin/env python
"""Local-only v2 collector. All mutating commands acquire the same state lock."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_v2.operations import daily, status, tick, verify, backup, restore_original_path
from runtime_v2.storage import lock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['verify','daily','commission','tick','status','backup','restore'])
    p.add_argument('--release',required=True)
    p.add_argument('--workspace',default='data/paper/research-v2')
    p.add_argument('--backup-path')
    p.add_argument('--archive')
    args=p.parse_args()
    if args.command=='verify': result=verify(args.release)
    elif args.command=='status': result=status(args.release,args.workspace)
    elif args.command=='tick': result=tick(args.release,args.workspace)
    elif args.command=='daily': result=daily(args.release,args.workspace,mode='manual')
    elif args.command=='commission': result=daily(args.release,str(Path(args.workspace)/'commissioning'),mode='commissioning')
    elif args.command=='restore':
        verify(args.release)
        if not args.backup_path or not args.archive: p.error('restore requires --backup-path and --archive; stop scheduler first')
        result=restore_original_path(args.backup_path,args.workspace,args.archive)
    else:
        verify(args.release)
        from datetime import datetime,timezone
        with lock(args.workspace): result=backup(args.workspace,datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    print(json.dumps(result,indent=2,default=str))

if __name__=='__main__': main()
