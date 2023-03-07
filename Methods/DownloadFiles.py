from urllib.request import urlretrieve
import os
import urllib.request

def download_zip_file(file_link: str, output_file: str='file.csv'):
        """
        Downloads a zip file from a link and saves it to the downloads folder.
    
        Parameters
        ----------
        file_link : str
            The link to the file to be downloaded.
        output_file : str
            A string containing the name of the file to be saved.
        
        Returns
        -------
        nothing
        """
    
        if not os.path.exists("downloads"):
            os.makedirs("downloads")
    
        try:
            if not os.path.exists(output_file):
                req = urllib.request.Request(
                    file_link,
                    data=None,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
                    }
                )
                with urllib.request.urlopen(req) as response, open(output_file, 'wb') as out_file:
                    out_file.write(response.read())
                print("File downloaded successfully")
            else:
                print("File already exists, skipping download")
        except Exception as e:
            print(f"Error downloading file: {e}")