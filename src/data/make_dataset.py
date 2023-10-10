import zipfile
from pathlib import Path

import kaggle

COMPETITION = 'airbus-ship-detection'

def download_dataset(out_path, small=False):
    out_path = Path(out_path)
    if small:
        # Small number of files that characterizes the competition
        files = [str(file) for file in kaggle.api.competition_list_files(COMPETITION)]

        for file in files:
            # Create directory for files
            fdir = (out_path / file).parent
            if not fdir.exists(): fdir.mkdir(parents=True)

            # Download file into the created directory
            kaggle.api.competition_download_file(COMPETITION, file, path=fdir)
    else:
        kaggle.api.competition_download_files(COMPETITION, out_path)

    # Extract all zip files and delete them
    zip_files = list(out_path.glob('*.zip'))
    for file in zip_files:
        with zipfile.ZipFile(file) as zip:
            zip.extractall(out_path)
        file.unlink()

if __name__ == '__main__':
    import sys

    dataset_size = sys.argv[1] if 2 <= len(sys.argv) else None
    if dataset_size is None or dataset_size not in ('original', 'small'):
        print('Usage: python make_dataset.py {original,small}')
        print('\toriginal: downloads the whole dataset of the competition')
        print('\tsmall: downloads a small dataset of the competition')
        exit(1)

    out_path = Path(__file__).parents[2] / 'data' / 'raw'
    if not out_path.exists():
        out_path.mkdir(parents=True)
    download_dataset(out_path, small=True)
