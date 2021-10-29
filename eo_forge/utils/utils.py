"""
General helper functions
========================

.. autosummary::
    :toctree: ../generated/

    rem_trail_os_sep
    walk_dir_files
"""
import fnmatch
import glob
import os


def rem_trail_os_sep(scene_path_dir):
    scene_path_dir_ = (
        scene_path_dir[:-1] if scene_path_dir.endswith(os.path.sep) else scene_path_dir
    )
    return scene_path_dir_


def walk_dir_files(target_base_path_, cases=("*.i", "*.o", "*.r")):
    """
    Get dir and files from a target_path.

    Parameters
    ----------
    target_base_path_: base path that will be check by os.walk
    cases: wild card extension to be checked in files (generates a dictionary)
    """
    IO_files_ = []
    IO_dirs_ = []
    for root, dirs, files in os.walk(target_base_path_):
        for name in files:
            IO_files_.append(os.path.join(root, name))
        for d in dirs:
            IO_dirs_.append(os.path.join(root, d))
        #
    IO_dict = {}
    #
    for c_ in cases:
        aux = []
        for d in IO_dirs_:
            aux.extend(glob.glob(os.path.join(d, c_)))
        for f in IO_files_:
            if fnmatch.fnmatch(f, c_):
                aux.extend([f])
        aux = list(dict.fromkeys(aux))
        IO_dict.update({c_: aux})

    return IO_files_, IO_dirs_, IO_dict
