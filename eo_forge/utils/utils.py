import os


def rem_trail_os_sep(scene_path_dir):
    scene_path_dir_ = (
        scene_path_dir[:-1] if scene_path_dir.endswith(os.path.sep) else scene_path_dir
    )
    return scene_path_dir_
