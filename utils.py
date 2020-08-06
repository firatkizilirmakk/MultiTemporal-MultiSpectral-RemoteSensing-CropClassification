import os
import sys
import urllib
import zipfile

from tqdm import tqdm


def update_progress(progress):
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rLoaded: [{0}] {1:.2f}% {2}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, overwrite=False):
    if url is None:
        raise ValueError("download_file: provided url is None!")

    if not os.path.exists(output_path) or overwrite:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    else:
        print(f"file exists in {output_path}. specify overwrite=True if intended")


def unzip(zipfile_path, target_dir):
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
