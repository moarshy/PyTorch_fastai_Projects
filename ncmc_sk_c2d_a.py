import os
import sys
import json


SEED=101
random.seed(SEED)
set_seed(SEED, True)

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
        print('ls', f'/data/inputs/{did}/0')
        print('ls2', os.listdir(f'/data/inputs/'))
        print('ls3', os.listdir(f'/data/ddos/'))

        print(f"Reading asset file {filename}.")
        print('type', type(os.listdir(f'/data/inputs/{did}/0/')[0]))
        filename = Path('/data/')  # 0 for metadata service

        return filename, did


def get_df(
    local:bool, # Flag to indicate local vs C2D
):
    print("Preparing df.")
    filename, did = get_input(local)
    # image_fns = get_image_files(filename)

    image_fns = []
    for root, dirs, files in os.walk(str(filename)):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            fn = os.path.join(root,file)
            if fn.split('.')[-1] in ['jpeg', 'jpg', 'png']:
                image_fns.append(Path(fn))
            print(len(path) * '---', file)

    print(f"Printing samples of image filenames: {image_fns[:3]}")
    df = pd.DataFrame(list(image_fns), columns=['fns'])

    df['label'] = df['fns'].apply(lambda x: get_label(x))
    df['patient_id'] = df['fns'].apply(lambda x: get_patient(x))
    df['is_valid'] = False

    df = get_train_test(df)

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


if __name__ == "__main__":
    print(f"Is cuda available: {torch.cuda.is_available()}")

    local = (len(sys.argv) == 2 and sys.argv[1] == "local")
    run(local)
