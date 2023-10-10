import csv
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


def filter_missing(annotations_file, imgs_dir):
    rows, fieldnames = [], []
    with open(annotations_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        imgs_dir = Path(imgs_dir)
        rows = [row for row in reader if (imgs_dir / row['ImageId']).exists()]

    with open(annotations_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    import sys

    dataset_size = sys.argv[1] if 2 <= len(sys.argv) else 'small'
    if dataset_size not in ('original', 'small'):
        print('Usage: python make_dataset.py [original,small]')
        print('\toriginal: downloads the whole dataset of the competition')
        print('\tsmall: downloads a small dataset of the competition')
        exit(1)

    out_path = Path(__file__).parents[2] / 'data' / 'raw'
    if not out_path.exists():
        out_path.mkdir(parents=True)

    download_dataset(out_path, small=dataset_size == 'small')

    print('Removing unused values from segmentations')
    filter_missing(out_path / 'train_ship_segmentations_v2.csv', out_path / 'train_v2')
