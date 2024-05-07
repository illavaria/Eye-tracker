from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.discovery import build
import pprint
import io


class GoogleCloudStorage(object):
    def __init__(self):
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.service_account_file = '/Users/illaria/BSUIR/Diploma/diploma-422409-aae94f436a83.json'
        self.credentials = service_account.Credentials.from_service_account_file(
                self.service_account_file, scopes=self.scopes)
        self.service = build('drive', 'v3', credentials=self.credentials)
        self.folder_id = '1eFQU5h7EMiAQFEMx-m6H7WYFG4ttPfh1'
        self.pp = pprint.PrettyPrinter(indent=4)

    def get_versions(self):
        query = f"'{self.folder_id}' in parents"
        results = self.service.files().list(q=query, pageSize=10, fields="nextPageToken, files(id, name)").execute()
        files = results.get('files', [])
        sorted_files = sorted(files, key=lambda x: x['name'], reverse=True)
        return sorted_files

    @staticmethod
    def get_versions_names(results):
        return [result['name'] for result in results]

    # @staticmethod
    # def get_folder_id_by_name(results, folder_name):
    #     folders = results.get('files', [])
    #     for folder in folders:
    #         if folder['name'] == folder_name:
    #             return folder['id']


    def print(self, data):
        self.pp.pprint(data)

    def upload_file(self, file_path, gcs_folder_id, gcs_file_name):
        # folder_id = '1ZlhoGArT_g_AvtoDsASgCpjlB9r657em'
        # name = 'epoch_1_left.pth'
        # file_path = '/Users/illaria/BSUIR/Diploma/code/PyTorchTry1/eyes_net_left/epoch_1.pth'
        file_metadata = {
                        'name': gcs_file_name,
                        'parents': [gcs_folder_id]
                    }
        media = MediaFileUpload(file_path, resumable=True)
        r = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        self.print(r)

    def download_file(self, gcs_file_id, file_path):
        # file_id = '1NAriMEwQZn7jLNLBeY3hdLcBo1QIvHOF'
        request = self.service.files().get_media(fileId=gcs_file_id)
        # filename = 'annotationscopy_google.txt'
        fh = io.FileIO(file_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("Download %d%%." % int(status.progress() * 100))

    def download_files_from_folder(self, folder_id):
        query = f"'{folder_id}' in parents"
        results = self.service.files().list(q=query, pageSize=100, fields="nextPageToken, files(id, name)").execute()
        files = results.get('files', [])

        for file in files:
            print(f"Downloading {file['name']}...")
            self.download_file(file['id'], file['name'])
            print(f"Downloaded {file['name']} successfully.")
        return True



cloud_manager = GoogleCloudStorage()
versions = cloud_manager.get_versions()
# versions = GoogleCloudStorage.get_versions_names(versions)
print(versions)




# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/illaria/BSUIR/Diploma/diploma-422409-aae94f436a83.json"
#
#
# class GoogleCloudStorageManager:
#     def __init__(self, bucket_name):
#         self.client = storage.Client()
#         # if self.client.bucket(bucket_name).exists():
#         #     self.bucket = self.client.bucket(bucket_name)
#         # else:
#         bucket = self.client.bucket(bucket_name)
#         bucket.storage_class = 'STANDARD'
#
#         bucket = self.client.create_bucket(bucket)
#         # self.client.create_bucket(bucket_name)
#         print(f'Bucket {bucket.name} successfully created.')
#
#     def upload_model(self, model_file_path, destination_blob_name):
#         """Загружает модель в GCS"""
#         blob = self.bucket.blob(destination_blob_name)
#         blob.upload_from_filename(model_file_path)
#         print(f"Model {model_file_path} uploaded to {destination_blob_name}.")
#
#     def download_model(self, blob_name, destination_file_path):
#         """Скачивает модель из GCS"""
#         blob = self.bucket.blob(blob_name)
#         blob.download_to_filename(destination_file_path)
#         print(f"Model {blob_name} downloaded to {destination_file_path}.")
#
#     def list_models(self):
#         """Возвращает список моделей в бакете"""
#         blobs = self.bucket.list_blobs()
#         models = [blob.name for blob in blobs]
#         print("Models in the bucket:")
#         for model in models:
#             print(model)
#         return models
#
#
# gcsmanager = GoogleCloudStorageManager(bucket_name="bucket_try1")
#
