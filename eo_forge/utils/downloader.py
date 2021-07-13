import os
import sys
from subprocess import run
import fnmatch
import shutil
from tempfile import mkdtemp
from subprocess import run
import tqdm
import pandas as pd
import queue
import threading
import warnings

from eo_forge.utils.landsat import (
    LANDSAT5_BANDS_RESOLUTION,
    LANDSAT8_BANDS_RESOLUTION,
)

from eo_forge.utils.sentinel import SENTINEL2_BANDS_RESOLUTION
from eo_forge.utils.utils import rem_trail_os_sep

# libs
from .logger import update_logger


class bucket_downloader(object):
    def __init__(self, logger=None):
        """
        :param logger=None
        """
        self.logger_ = logger

    @staticmethod
    def gc_bucket_download(data_path, data_base):
        """
        :param
        """
        cmd = ["gsutil", "-m", "cp", "-r", data_path, data_base]
        p = run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            stdout = p.stdout
            stderr = ""
        else:
            stdout = p.stdout
            stderr = p.stderr
        return stdout, stderr

    def bucket_download(
        self, data_path_list, data_base, force_download=False, verbose=False
    ):
        """
        :param
        """
        for d in tqdm.tqdm(data_path_list, desc="Files "):
            # check if already exists
            d_basename = os.path.basename(d)
            if os.path.isdir(os.path.join(data_base, d_basename)):
                if force_download:
                    update_logger(
                        self.logger_,
                        "Force Download of {} (dir already existed) ".format(d),
                        "WARNING",
                    )
                    stdout, stderr = self.gc_bucket_download(d, data_base)
                    if stderr:
                        update_logger(
                            self.logger_,
                            "FAILED @ {} with error {}".format(d, stderr),
                            "ERROR",
                        )
                    if stdout:
                        update_logger(
                            self.logger_,
                            "OK @ {} with msg {}".format(d, stdout),
                            "INFO",
                        )
                else:
                    update_logger(
                        self.logger_,
                        "Dir {} already existed".format(d_basename),
                        "INFO",
                    )
            else:
                update_logger(
                    self.logger_,
                    "Downloading {} ".format(d_basename),
                    "INFO",
                )
                stdout, stderr = self.gc_bucket_download(d, data_base)
                if stderr:
                    update_logger(
                        self.logger_,
                        "FAILED @ {} with error {}".format(d, stderr),
                        "ERROR",
                    )
                if stdout:
                    update_logger(
                        self.logger_,
                        "OK @ {} with msg {}".format(d, stdout),
                        "INFO",
                    )


def check_clouds(file):
    with open(file, "r") as f:
        lines = f.readlines()
    cloud = None
    for line in lines:
        if "CLOUD_COVER_LAND" in line:
            cloud = line.split("=")[1].replace("\n", "").strip()
    return cloud


class gcSatImg(object):

    BASE_LANDSAT8 = "gs://gcp-public-data-landsat/LC08/01/{}/{}/"
    BASE_LANDSAT5 = "gs://gcp-public-data-landsat/LT05/01/{}/{}/"
    BASE_SENTINEL2 = ""
    DATE_LANDSAT8 = 3
    DATE_LANDSAT5 = 3
    DATE_SENTINEL2 = None

    def __init__(self, sat="L8"):
        if sat == "L8":
            self.base_url = self.BASE_LANDSAT8
            self.date_idx = self.DATE_LANDSAT8
        elif sat == "L5":
            self.base_url = self.BASE_LANDSAT5
            self.date_idx = self.DATE_LANDSAT5
        else:
            raise ("EITHER L8 or L5")

    def gcImagesCheck(self, path, row):
        BASE_SAT = self.base_url
        URL = BASE_SAT.format(path, row)
        cmd = ["gsutil", "ls", URL]
        p = run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            stdout = p.stdout
            stderr = ""
        else:
            stdout = p.stdout
            stderr = p.stderr

        if stdout and p.returncode == 0:
            self.sat_imgs_flag = True
            self.sat_imgs = stdout.split("\n")
            self.sat_imgs_err = stderr
        else:
            self.sat_imgs_flag = False
            self.sat_imgs = None
            self.sat_imgs_err = stderr

    @staticmethod
    def get_clouds_from_meta(pd_filt, meta_key="_MTL.txt"):
        """
        :param
        """
        dir_ = mkdtemp()
        cloud_ = []
        for i, r in pd_filt.iterrows():
            prod = r["product-id"]
            prod_path = r["base-url"]
            data_path = (os.path.join(prod_path, prod) + "{}").format(meta_key)
            data_base = dir_
            cmd = ["gsutil", "cp", data_path, data_base]
            p = run(cmd, capture_output=True, text=True)
            if p.returncode == 0:
                file_mtl = (os.path.join(data_base, prod) + "{}").format(meta_key)
                cloud_.append(check_clouds(file_mtl))
            else:
                print(f"FAIL @ {data_path}")
                cloud_.append(None)
        shutil.rmtree(dir_)
        pd_filt["clouds"] = cloud_
        return pd_filt

    def gcImagesFilt(
        self,
        filters=["*_L1TP_", "*_T1"],
        dates=[None, None],
        clouds=True,
        meta_key="_MTL.txt",
    ):
        """"""
        filtered_imgs = []
        if self.sat_imgs_flag:
            for scene_path_dir in self.sat_imgs:
                scene_path_dir = rem_trail_os_sep(scene_path_dir)
                #
                product_id = os.path.basename(scene_path_dir)
                pattern = "".join(filters)
                if fnmatch.fnmatch(product_id, pattern):
                    filtered_imgs.append([product_id, scene_path_dir])

            pd_ = pd.DataFrame(filtered_imgs, columns=["product-id", "base-url"])
            pd_["date"] = pd.to_datetime(
                pd_["product-id"].apply(lambda x: x.split("_")[self.date_idx]),
                format="%Y%m%d",
            )
            # filt dates
            if dates[0]:
                pd_min = pd_[pd_["date"] >= dates[0]]
            else:
                pd_min = pd_

            if dates[1]:
                pd_max = pd_min[pd_min["date"] <= dates[1]]
            else:
                pd_max = pd_min

            pd_ = pd_max.copy()
            pd_.reset_index(inplace=True, drop=True)
            if clouds:
                pd_ = self.get_clouds_from_meta(pd_, meta_key=meta_key)
        else:
            pd_ = pd.DataFrame()
        #
        self.filt_imgs = filtered_imgs
        self.pd_filt = pd_


class ThreadUrlDownload(threading.Thread):
    """Threaded Url download"""

    def __init__(self, queue_):
        threading.Thread.__init__(self)
        self.queue = queue_

    def run(self):

        while True:
            # grabs cmd from queue
            cmdi = self.queue.get()
            #
            p = run(cmdi, capture_output=True, text=True)
            #
            # signals to queue job is done
            self.queue.task_done()


class bucket_images_downloader(object):
    """
    Base class to download images
    """

    def __init__(self, spacecraft="L8", bands=None, logger=None):
        """"""

        if spacecraft not in ("L5", "L8", "S2"):
            raise ValueError(
                f"Only Landsat5 (L5) , Landsat8 (L8) and Sentinel2 (S2) are supported. "
                f'Spacecraft received: "{spacecraft}"'
            )
        self.spacecraft = spacecraft

        self.logger_ = logger
        if spacecraft == "L8":
            self._ordered_bands = tuple(LANDSAT8_BANDS_RESOLUTION.keys())
            self._extra_bands = ["BQA"]
            self._meta_key = ["MTL.txt"]
        elif spacecraft == "L5":
            self._ordered_bands = tuple(LANDSAT5_BANDS_RESOLUTION.keys())
            self._extra_bands = ["BQA"]
            self._meta_key = ["MTL.txt"]
        elif spacecraft == "S2":
            self._ordered_bands = tuple(SENTINEL2_BANDS_RESOLUTION.keys())
            self._meta_key = ["MTD_MSIL1C.xml"]
            raise ValueError(f" Sentinel2 WIP")

        if bands is None:
            self.bands = self._ordered_bands
        else:
            self.bands = []
            for band in bands:
                if band not in self._ordered_bands:
                    warnings.warn(f"'{band}' is not a valid band. Ignoring")
                else:
                    self.bands.append(band)

    def build_datapath_landsat(self, bucket_list=[], bucket_arxive=[], bqa_clouds=True):
        """"""
        bucket_atomic = []
        bucket_atomic_arxive = []
        for bi, ba in zip(bucket_list, bucket_arxive):
            bi = rem_trail_os_sep(bi)
            bi_base = os.path.basename(bi)
            for b in self.bands:
                bucket_atomic.append(os.path.join(bi, bi_base + "_" + b + ".TIF"))
                bucket_atomic_arxive.append(ba)
            if bqa_clouds:
                for b in self._extra_bands:
                    bucket_atomic.append(os.path.join(bi, bi_base + "_" + b + ".TIF"))
                    bucket_atomic_arxive.append(ba)
            bucket_atomic.append(os.path.join(bi, bi_base + "_MTL.txt"))
            bucket_atomic_arxive.append(ba)

        return bucket_atomic, bucket_atomic_arxive

    def execute(
        self,
        bucket_cases=[],
        arxive="./",
        bqa_clouds=True,
        max_proc_thread=1,
        force_download=False,
    ):
        """"""
        # build queue

        bucket_cases_proc = []
        bucket_arxive_proc = []
        for bki in bucket_cases:
            bki = rem_trail_os_sep(bki)
            bki_base = os.path.basename(bki)
            arxive_i = os.path.join(arxive, bki_base)
            if not os.path.isdir(arxive_i):
                os.makedirs(arxive_i)
                bucket_cases_proc.append(bki)
                bucket_arxive_proc.append(arxive_i)
            elif os.path.isdir(arxive_i) and force_download:
                bucket_cases_proc.append(bki)
                bucket_arxive_proc.append(arxive_i)
                update_logger(
                    self.logger_,
                    "Force Download of {} (dir already existed) ".format(arxive_i),
                    "WARNING",
                )
            else:
                pass

        if self.spacecraft in ("L5", "L8"):
            data_path_atomic, data_base_atomic = self.build_datapath_landsat(
                bucket_cases_proc, bucket_arxive_proc, bqa_clouds
            )

        i_cases = []
        for dbk, abk in zip(data_path_atomic, data_base_atomic):
            cmd = ["gsutil", "cp", dbk, abk]
            i_cases.append(cmd)

        q = queue.Queue(maxsize=len(i_cases))
        for qc in i_cases:
            q.put(qc)

        for i in range(max_proc_thread):
            t = ThreadUrlDownload(q)
            t.setDaemon(True)
            t.start()
        q.join()

