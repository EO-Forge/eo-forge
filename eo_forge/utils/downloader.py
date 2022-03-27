"""
Downloader module
==================

.. autosummary::
    :toctree: ../generated/

    bucket_downloader
    gcSatImg
    ThreadUrlDownload
    bucket_images_downloader
"""

import fnmatch

import os
import pandas as pd
import platform
import queue
import shutil
import threading
import tqdm
import warnings
from pathlib import Path
from subprocess import run
from tempfile import mkdtemp

from eo_forge import check_logger
from eo_forge.utils.landsat import (
    LANDSAT5_BANDS_RESOLUTION,
    LANDSAT8_BANDS_RESOLUTION,
    get_clouds_landsat,
)
from eo_forge.utils.sentinel import (
    SENTINEL2_BANDS_RESOLUTION,
    get_clouds_msil1c,
    get_granule_from_meta_sentinel,
)
from eo_forge.utils.utils import rem_trail_os_sep


class bucket_downloader:
    """
    Downloader Sentinel or Landasat data from a Google Cloud bucket.
    """

    def __init__(self, logger=None):
        """
        Constructor

        Parameters
        ----------
        logger: None or logging module instance
            logger to populate (if provided). If None, a default logger is used that
            print all the messages to the stdout.
        """
        self.logger = check_logger(logger)

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
                    self.logger.warning(f"Forced download of {d} (dir already existed)")
                    stdout, stderr = self.gc_bucket_download(d, data_base)
                    if stderr:
                        self.logger.error(f"FAILED @ {d} with error {stderr}")
                    if stdout:
                        self.logger.info(f"OK @ {d} with msg {stdout}")
                else:
                    self.logger.info(f"Dir {d_basename} already existed")
            else:
                self.logger.info("Downloading {} ".format(d_basename))
                stdout, stderr = self.gc_bucket_download(d, data_base)
                if stderr:
                    self.logger.error("FAILED @ {} with error {}".format(d, stderr))
                if stdout:
                    self.logger.info("OK @ {} with msg {}".format(d, stdout))


class gcSatImg:
    """
    Google cloud definitions and path generators.
    """

    BASE_LANDSAT8 = "gs://gcp-public-data-landsat/LC08/01/{}/{}/"
    BASE_LANDSAT5 = "gs://gcp-public-data-landsat/LT05/01/{}/{}/"
    DATE_LANDSAT8 = 3
    DATE_LANDSAT5 = 3
    LANDSAT_FILTERS = ["*_L1TP_", "*_T1"]
    LANDSAT_META = "{}/{}_MTL.txt"
    #
    BASE_SENTINEL2 = "gs://gcp-public-data-sentinel-2/tiles/{}/{}/{}/"
    DATE_SENTINEL2 = 2
    SENTINEL_FILTERS = []
    SENTINEL2_META = "{}/MTD_MSIL1C.xml"

    def __init__(self, spacecraft="L8", boto_config=None, logger=None):
        """
        Initialize the google cloud image checker

        Parameters
        ----------
        sat: str
            sat acronym: L8,L5, S2
        boto_config: Path or None
            path to BOTO_CONFIG file. If None will try to
            use $HOME/.boto as location
        logger: None or logging module instance
            logger to populate (if provided). If None, a default logger is used that
            print all the messages to the stdout.
        """

        self.logger = check_logger(logger)

        if spacecraft == "L8":
            self.base_url = self.BASE_LANDSAT8
            self.date_idx = self.DATE_LANDSAT8
            self.clouds_from_meta = get_clouds_landsat
        elif spacecraft == "L5":
            self.base_url = self.BASE_LANDSAT5
            self.date_idx = self.DATE_LANDSAT5
            self.clouds_from_meta = get_clouds_landsat
        elif spacecraft == "S2":
            self.base_url = self.BASE_SENTINEL2
            self.date_idx = self.DATE_SENTINEL2
            self.clouds_from_meta = get_clouds_msil1c
        else:
            raise ValueError("EITHER L8,L5,S2")

        self.spacecraft = spacecraft
        self.logger.info(f"Running on spacecraft {self.spacecraft}")
        if boto_config:
            os.environ["BOTO_CONFIG"] = boto_config
            self.boto_path = os.getenv("BOTO_CONFIG")
        else:
            if platform.system() != "Windows":
                self.boto_path = Path(os.getenv("HOME")) / ".boto"
            else:
                self.boto_path = Path(os.getenv("USERPROFILE")) / ".boto"
        self.logger.info(f"Setting boto path to: {self.boto_path}")

    def gcImagesCheck(self, url_filler):
        """
        gsutil command assembler for image checking.

        Parameters
        -----------
        url_filler: list
            list with parameter for platforms,
            landsat -> [path: str, row:str]
            sentinel2 -> [UTM_ZONE:str, LATITUDE_BAND:str, GRID_SQUARE:str]

        Returns
        -------
        None

        """
        BASE_SAT = self.base_url
        URL = BASE_SAT.format(*url_filler)
        cmd = ["gsutil", "ls", URL]

        self.logger.debug(f"Running: {' '.join(cmd)}")
        p = run(cmd, capture_output=True, text=True)

        self.logger.info(f"Checking bucket: {' '.join(cmd)}")
        if p.returncode == 0:
            stdout = p.stdout
            stderr = ""
        else:
            stdout = p.stdout
            stderr = p.stderr
            self.logger.error(f"gsutil returned a non-zero exit code.")
            self.logger.error(f"gsutil error: {stderr}")

            if "ServiceException: 401" in stderr + stdout:
                self.logger.error(
                    f"It seems that you need to authorize gsutil to access Google Cloud.\n"
                    'Try running "gsutil config"'
                )

        if stdout and p.returncode == 0:
            self.sat_imgs_flag = True
            self.sat_imgs = stdout.split("\n")
            self.sat_imgs_err = stderr
        else:
            self.sat_imgs_flag = False
            self.sat_imgs = None
            self.sat_imgs_err = stderr

    def build_metadata_path(self, base_path, prod_id):
        """Build metadata path based on platform."""
        if self.spacecraft != "S2":
            data_path = self.LANDSAT_META.format(base_path, prod_id)
        else:
            data_path = self.SENTINEL2_META.format(base_path)

        return data_path

    def get_clouds_from_metadata(self, file):
        """
        Get clouds coverage based on platform/

        Parameters
        ----------
        file: path
            path to file

        Returns
        -------
        cloud level
        """
        if self.spacecraft != "S2":
            clouds = get_clouds_landsat(file)
        else:
            clouds = get_clouds_msil1c(file)

        return clouds

    def get_clouds_from_meta(self, pd_filt, remove_meta=True):
        """
        Get cloud file from metadata.
        """
        dir_ = mkdtemp()
        cloud_ = []
        for i, r in pd_filt.iterrows():
            prod_id = r["product-id"]
            bucket_path = r["base-url"]
            local_path = dir_  # to dump files
            remote_path = self.build_metadata_path(bucket_path, prod_id)

            cmd = ["gsutil", "cp", remote_path, local_path]
            self.logger.debug(f"Copying meta with cmd: {' '.join(cmd)}")
            p = run(cmd, capture_output=True, text=True)
            if p.returncode == 0:
                file_metadata = self.build_metadata_path(local_path, prod_id)
                cloud_.append(self.clouds_from_meta(file_metadata))
                #
                if remove_meta:
                    os.remove(file_metadata)
            else:
                cloud_.append(None)
                self.logger.warning(
                    f"FAIL @ {remote_path} appending None as cloud level"
                )
        shutil.rmtree(dir_)
        pd_filt["clouds"] = cloud_
        return pd_filt

    def make_dates_from_name(self, pd_, date_col="date"):
        """Generate dates from files name.

        Parameters
        ----------
        pd_: pandas Dataframe
            Pandas dataframe with file list.

        Returns
        -------
        pd_: pandas dataframe
            Updated dataframe
        """

        def split_get_datestr(x, x_idx, x_in_idx=None, x_splitter="_"):
            """"""
            if x_in_idx is None:
                return x.split(x_splitter)[x_idx]
            else:
                x_tmp = x.split(x_splitter)[x_idx]
                return x_tmp[:x_in_idx]

        if self.spacecraft != "S2":
            pd_[date_col] = pd.to_datetime(
                pd_["product-id"].apply(lambda x: split_get_datestr(x, self.date_idx)),
                format="%Y%m%d",
            )
        else:
            pd_[date_col] = pd.to_datetime(
                pd_["product-id"].apply(
                    lambda x: split_get_datestr(x, self.date_idx, 8)
                ),
                format="%Y%m%d",
            )

        return pd_

    @staticmethod
    def filt_dates(pd_, dates=(None, None), date_col="date"):
        """
        Filter dates on dataframe.

        Parameters
        ----------
        pd_: pandas dataframe
            dataframe instance with data to filt
        dates: list
            if [None,None] dataframe is returned as is
            else [yyyy-mm-dd,yyyy-mm-dd] is expected
        date_col: str
            date column name on pd_

        Returns
        -------
        pd_: pandas dataframe
            filtered dataframe (or original is dates is None)
        """
        # filt dates
        if dates[0]:
            pd_min = pd_[pd_[date_col] >= dates[0]]
        else:
            pd_min = pd_

        if dates[1]:
            pd_max = pd_min[pd_min[date_col] <= dates[1]]
        else:
            pd_max = pd_min

        pd_ = pd_max.copy()
        pd_.reset_index(inplace=True, drop=True)
        return pd_

    @staticmethod
    def clean_scene_name(scene_path_dir):
        """Clean scene name."""
        # generic
        scene_path_dir = rem_trail_os_sep(scene_path_dir)
        # sentinel
        scene_dir_cleaned = scene_path_dir.replace("_$folder$", "")
        #
        return scene_dir_cleaned

    @staticmethod
    def clean_dataframe_values(pd_):
        """Remove NaN values from dataframe."""
        nan_value = float("NaN")
        pd_.replace("", nan_value, inplace=True)
        pd_.dropna(inplace=True)
        pd_ret = pd_[~pd_.duplicated()]
        return pd_ret

    def gcImagesFilt(
        self,
        filters=[],
        dates=[None, None],
        clouds=True,
    ):
        """Process images metadata obtained from gc bucket.

        Parameters
        ----------
        filters: list
            Filters to use. No filters are applied if filters=[] (empty list) or None.
            Otherwise, the filters are applied on the names.
            E.g., filters=[key1,key2,etc] is joined as ''.join([key1,key2,etc]) and then
            used as pattern by fnmatch.fnmatch.
        dates: list
            If [None,None] data is returned as it is.
            Else [yyyy-mm-dd,yyyy-mm-dd] is expected
        clouds: bool
            flag to try to obtain clouds from metadata
        """
        if filters is None:
            filters = []
        if dates is None:
            dates = [None, None]

        filtered_imgs = []
        if self.sat_imgs_flag:
            for scene_path_dir in self.sat_imgs:

                # clean scene path
                scene_path_dir = self.clean_scene_name(scene_path_dir)
                #
                product_id = os.path.basename(scene_path_dir)

                if filters:
                    pattern = "".join(filters)
                    if fnmatch.fnmatch(product_id, pattern):
                        filtered_imgs.append([product_id, scene_path_dir])
                else:
                    filtered_imgs.append([product_id, scene_path_dir])

            pd_ = pd.DataFrame(filtered_imgs, columns=["product-id", "base-url"])
            # clean pd from empty values
            pd_ = self.clean_dataframe_values(pd_)
            # get dates
            pd_ = self.make_dates_from_name(pd_, date_col="date")
            #
            # filt dates
            # at this point we need a "date" column
            pd_ = self.filt_dates(pd_, dates=dates, date_col="date")
            # clouds
            if clouds:
                pd_ = self.get_clouds_from_meta(pd_)
        else:
            if clouds:
                pd_ = pd.DataFrame(columns=["product-id", "base-url", "date", "clouds"])
            else:
                pd_ = pd.DataFrame(columns=["product-id", "base-url", "date"])
        #
        self.filt_imgs = filtered_imgs
        self.pd_filt = pd_


class ThreadUrlDownload(threading.Thread):
    """Threaded Url download."""

    def __init__(self, queue_,logger=None):
        threading.Thread.__init__(self)
        self.queue = queue_
        self.logger_=logger

    def run(self):

        while True:
            # grabs cmd from queue
            cmdi = self.queue.get()
            #
            p = run(cmdi, capture_output=True, text=True)
            if self.logger_:
                if p.returncode == 0:
                    self.logger_.info(f'{cmdi} - OK')
                else:
                    self.logger_.warning(f'{cmdi} - FAIL')
            #
            # signals to queue job is done
            self.queue.task_done()


class bucket_images_downloader:
    """
    Base class to download images.
    """

    def __init__(self, spacecraft="L8", bands=None, logger=None):
        """
        Constructor.

        Parameters
        ----------
        spacecraft: int
            Landsat spacecraft (5 or 8).
        bands: iterable
            List of bands to process.
        logger: None or logging module instance
            logger to populate (if provided). If None, a default logger is used that
            print all the messages to the stdout.
        """

        self.logger = check_logger(logger)

        if spacecraft not in ("L5", "L8", "S2"):
            raise ValueError(
                f"Only Landsat5 (L5) , Landsat8 (L8) and Sentinel2 (S2) are supported. "
                f'Spacecraft received: "{spacecraft}"'
            )
        self.spacecraft = spacecraft
        self.logger.info(f"Running on spacecraft {self.spacecraft}")

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

        if bands is None:
            self.bands = self._ordered_bands
        else:
            self.bands = []
            for band in bands:
                if band not in self._ordered_bands:
                    warnings.warn(f"'{band}' is not a valid band. Ignoring")
                else:
                    self.bands.append(band)

        self.logger.info(f"Requesting bands {self.bands}")

    def build_datapath_landsat(
        self, bucket_list=[], bucket_archive=[], bqa_clouds=True
    ):
        bucket_atomic = []
        bucket_atomic_archive = []
        for bi, ba in zip(bucket_list, bucket_archive):
            bi = rem_trail_os_sep(bi)
            bi_base = os.path.basename(bi)
            for b in self.bands:
                bucket_path = f"{bi}/{bi_base}_{b}.TIF"
                bucket_atomic.append(bucket_path)
                bucket_atomic_archive.append(ba)
            if bqa_clouds:
                for b in self._extra_bands:
                    bucket_path = f"{bi}/{bi_base}_{b}.TIF"
                    bucket_atomic.append(bucket_path)
                    bucket_atomic_archive.append(ba)
            bucket_atomic.append(f"{bi}/{bi_base}_MTL.txt")
            bucket_atomic_archive.append(ba)

        return bucket_atomic, bucket_atomic_archive

    def build_datapath_sentinel2(
        self, bucket_list=None, bucket_archive=None, bqa_clouds=True, keep_safe=True
    ):
        """Build datapath for Sentinel2.

        Parameters
        ----------
        bucket_list: list
            List of gcp url with images. Empty list by default.
        bucket_archive: list
            List with local path to download. Empty list by default.
        bqa_clouds: bool
            If cloud mas is required

        Notes
        -----
        PRODUCT_ID/GRANULE/{GRANULE_ID}/IMG_DATA/{IMAGE_BASE}_{BANDS}.jp2
        PRODUCT_ID/GRANULE/{GRANULE_ID}/QI_DATA/MSK_CLOUDS_B00.gml (<20220125)
        PRODUCT_ID/GRANULE/{GRANULE_ID}/QI_DATA/MSK_CLASSI_B00.jp2 (>=20220125)
        """
        if bucket_list is None:
            bucket_list = list()

        if bucket_archive is None:
            bucket_archive = list()

        SENTINEL2_URL_BANDS = "{}/GRANULE/{}/IMG_DATA/{}_{}.jp2"
        SENTINEL2_URL_CLOUDS = ["{}/GRANULE/{}/QI_DATA/MSK_CLOUDS_B00.gml","{}/GRANULE/{}/QI_DATA/MSK_CLASSI_B00.jp2"]
        SENTINEL2_META = "{}/*.xml"

        if keep_safe:
            SENTINEL2_LOCAL_BANDS = "{}/GRANULE/{}/IMG_DATA/"
            SENTINEL2_LOCAL_CLOUDS = "{}/GRANULE/{}/QI_DATA/"
        else:
            SENTINEL2_LOCAL_BANDS = "{}/"
            SENTINEL2_LOCAL_CLOUDS = "{}/"

        bucket_atomic = []
        bucket_atomic_archive = []
        for bi, ba in zip(bucket_list, bucket_archive):

            bi = rem_trail_os_sep(bi)

            # get metadatafile
            granulei, image_basei = get_granule_from_meta_sentinel(bi)
            for b in self.bands:
                bucket_atomic.append(
                    SENTINEL2_URL_BANDS.format(bi, granulei, image_basei, b)
                )
                #
                if keep_safe:
                    local_img_data = SENTINEL2_LOCAL_BANDS.format(ba, granulei)
                else:
                    local_img_data = SENTINEL2_LOCAL_BANDS.format(ba)

                bucket_atomic_archive.append(local_img_data)
                # check local dir
                if os.path.isdir(local_img_data):
                    pass
                else:
                    os.makedirs(local_img_data, exist_ok=True)

            if bqa_clouds:
                for mask in SENTINEL2_URL_CLOUDS:
                    bucket_atomic.append(mask.format(bi, granulei))

                    if keep_safe:
                        local_qi = SENTINEL2_LOCAL_CLOUDS.format(ba, granulei)
                    else:
                        local_qi = SENTINEL2_LOCAL_CLOUDS.format(ba)

                    bucket_atomic_archive.append(local_qi)

                    if os.path.isdir(local_qi):
                        pass
                    else:
                        os.makedirs(local_qi, exist_ok=True)

            bucket_atomic.append(SENTINEL2_META.format(bi))
            bucket_atomic_archive.append(ba)

        return bucket_atomic, bucket_atomic_archive

    def execute(
        self,
        bucket_cases=None,
        archive="./",
        bqa_clouds=True,
        max_proc_thread=1,
        force_download=False,
    ):
        if bucket_cases is None:
            bucket_cases = list()
        bucket_cases_proc = []
        bucket_archive_proc = []
        for bki in bucket_cases:
            bki = rem_trail_os_sep(bki)
            bki_base = os.path.basename(bki)
            archive_i = os.path.join(archive, bki_base)
            if not os.path.isdir(archive_i):
                os.makedirs(archive_i)
                bucket_cases_proc.append(bki)
                bucket_archive_proc.append(archive_i)
            else:
                if force_download:
                    bucket_cases_proc.append(bki)
                    bucket_archive_proc.append(archive_i)
                    self.logger.warning(
                        f"Force Download of {archive_i} (dir already existed)."
                    )
                else:
                    self.logger.info(f"Skipping {archive_i} (dir already existed).")

        if self.spacecraft in ("L5", "L8"):
            data_path_atomic, data_base_atomic = self.build_datapath_landsat(
                bucket_cases_proc, bucket_archive_proc, bqa_clouds
            )
        else:
            data_path_atomic, data_base_atomic = self.build_datapath_sentinel2(
                bucket_cases_proc, bucket_archive_proc, bqa_clouds
            )

        i_cases = []
        for dbk, abk in zip(data_path_atomic, data_base_atomic):
            cmd = ["gsutil", "cp", dbk, abk]
            i_cases.append(cmd)

        q = queue.Queue(maxsize=len(i_cases))
        for qc in i_cases:
            q.put(qc)
            self.logger.debug(f"Queueing cmd for download: {' '.join(qc)}")

        for _ in range(max_proc_thread):
            t = ThreadUrlDownload(q,self.logger)
            t.setDaemon(True)
            t.start()
        q.join()
