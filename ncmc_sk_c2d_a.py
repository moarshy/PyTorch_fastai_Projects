import os
import sys
import json
import pandas as pd
from pathlib import Path
from PIL import Image


def get_input(
    local:bool=False, # Flag to indicate local vs C2D
):
    if local:
        print("Reading local medicaldata directory.")
        # Root directory for dataset
        filename = Path('./data')
        return filename

    dids = os.getenv('DIDS', None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    cwd = os.getcwd()
    print('cwd', cwd)

    did = dids[0]
    print(f"DID: {did}")

    for did in dids:
        print('ls', Path(f'/data/inputs/{did}').ls())
        filename = Path(f'/data/inputs/{did}')

        return filename


def get_df(
    local:bool, # Flag to indicate local vs C2D
):
    print("Preparing df.")
    filename = get_input(local)

    results_dir = Path('results')
    if not results_dir.exists():
        results_dir.mkdir()

    with open(filename) as datafile:
        print(type(datafile))
        print(datafile)
        data = datafile.read()
        print(data)

    teal_images = sorted(list(filename.glob('*')))
    print(teal_images)


def setup_train(
    local:bool, # Flag to indicate local vs C2D
):
    df = get_df(local)


def run(
    local:bool=False
):
    if local:
        print(f"You are in local env")
    if not local:
        print(f"You are in C2D")

    setup_train(local)

if __name__ == "__main__":

    local = (len(sys.argv) == 2 and sys.argv[1] == "local")
    run(local)
