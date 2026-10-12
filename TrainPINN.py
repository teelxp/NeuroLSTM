#!/usr/bin/env python3
"""Отдельный запуск базовой PINN; нужны пространственно-временные данные."""
import argparse
import json
from neurolstm.pinn import train_pinn

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Газовая PINN по (6.4)-(6.7)")
    parser.add_argument("--config",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(train_pinn(args.config,args.output),ensure_ascii=False,indent=2))
