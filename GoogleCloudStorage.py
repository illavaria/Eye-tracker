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

    def print(self, data):
        self.pp.pprint(data)

    def upload_file(self, file_path, gcs_folder_id, gcs_file_name):
        file_metadata = {
                        'name': gcs_file_name,
                        'parents': [gcs_folder_id]
                    }
        media = MediaFileUpload(file_path, resumable=True)
        r = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        self.print(r)

    def download_file(self, gcs_file_id, file_path):
        request = self.service.files().get_media(fileId=gcs_file_id)
        fh = io.FileIO(file_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("Download %d%%." % int(status.progress() * 100))

    def download_files_from_folder(self, folder_id, folder_name):
        query = f"'{folder_id}' in parents"
        results = self.service.files().list(q=query, pageSize=100, fields="nextPageToken, files(id, name)").execute()
        files = results.get('files', [])

        for file in files:
            print(f"Downloading {file['name']}...")
            self.download_file(file['id'], str(folder_name + '_' + file['name']))
            print(f"Downloaded {file['name']} successfully.")
        return True



# cloud_manager = GoogleCloudStorage()
# versions = cloud_manager.get_versions()
# versions = GoogleCloudStorage.get_versions_names(versions)
# print(versions)



