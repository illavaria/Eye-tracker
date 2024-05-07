import os


class ModelFileManager:
    @staticmethod
    def get_latest_model_if_exists():
        files = os.listdir(os.curdir)
        files.sort(reverse=True)
        print(files)
        count_files = 0
        model_files = {}
        model_version = ''
        for file in files:
            if file.endswith("left.pth"):
                count_files += 1
                model_files['left'] = file
                model_version = file.removesuffix('_left.pth')
            elif file.endswith("right.pth"):
                count_files += 1
                model_files['right'] = file
            if count_files == 2:
                break
        return count_files == 2, model_version, model_files

    @staticmethod
    def get_loaded_versions():
        files = os.listdir(os.curdir)
        files.sort(reverse=True)
        versions = []
        for file in files:
            if file.endswith("left.pth"):
                versions.append(file.removesuffix('_left.pth'))
        return versions
