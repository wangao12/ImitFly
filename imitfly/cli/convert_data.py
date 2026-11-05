import argparse
from pathlib import Path
from imitfly.data import convert_legacy_episode

def main():
    p = argparse.ArgumentParser(description="Convert one historical dataset_N directory")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    print(f"Converted {convert_legacy_episode(a.source, a.output)} aligned frames")

if __name__ == "__main__":
    main()
