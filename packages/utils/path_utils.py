from pathlib import Path
from os import scandir
image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']


def get_image_paths(dir_path, image_extensions=image_extensions):
    dir_path = Path(dir_path)

    result = []
    if dir_path.exists():
        for x in list(scandir(str(dir_path))):
            if any([x.name.lower().endswith(ext) for ext in image_extensions]):
                result.append(x.path)
    return result
