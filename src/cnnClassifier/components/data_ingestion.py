import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
     
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        
        # Self-correction: Handle data reorganization for Keras flow_from_directory
        import pandas as pd
        import shutil

        def reorganize_split(split):
            base_dir = os.path.join(unzip_path, split)
            csv_path = os.path.join(base_dir, "_classes.csv")
            
            if not os.path.exists(csv_path):
                return

            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            
            for _, row in df.iterrows():
                filename = row['filename']
                src_path = os.path.join(base_dir, filename)
                
                if not os.path.exists(src_path):
                    continue
                    
                if row['normal'] == 1:
                    class_folder = "normal"
                else:
                    class_folder = "cancer"
                    
                dest_dir = os.path.join(base_dir, class_folder)
                os.makedirs(dest_dir, exist_ok=True)
                
                shutil.move(src_path, os.path.join(dest_dir, filename))

        for split in ["train", "valid", "test"]:
            reorganize_split(split)
        
        logger.info(f"Extracted and reorganized data in {unzip_path}")

        