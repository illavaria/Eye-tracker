import os


class ModelFileManager:
    @staticmethod
    def get_latest_model_if_exists():
        files = os.listdir(os.curdir)
        files.sort(reverse=True)
        print(files)
        model_files = {}
        model_version = ''
        for file in files:
            if file.endswith("right.pth"):
                model_files['right'] = file
                model_version = file.removesuffix('_right.pth')
            elif file.endswith("left.pth") and file.startswith(model_version):
                model_files['left'] = file
            if len(model_files) == 2:
                return True, model_version, model_files
        return False, '', {}

    @staticmethod
    def get_loaded_versions():
        files = os.listdir(os.curdir)
        files.sort(reverse=True)
        versions = []
        versions_hash = []
        for file in files:
            if file.endswith("right.pth"):
                versions_hash.append(file.removesuffix('_right.pth'))
            if file.endswith("left.pth") and file.removesuffix('_left.pth') in versions_hash:
                versions.append(file.removesuffix('_left.pth'))
        return versions
