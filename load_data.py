import os
import pandas as pd
import numpy as np
import logging
import ray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def load_dataset(file_path, header='infer', index_col=None):
    try:
        logger.info(f"Loading dataset from {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=header, index_col=index_col)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path, header=header, index_col=index_col)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
            
        logger.info(f"Successfully loaded dataset with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        return None
def preprocess_dataset(df, target_col=None, test_size=0.2, random_state=42, scale=False):
    try:
        if df is None or df.empty:
            logger.error("Cannot preprocess empty dataset")
            return None
            
        logger.info(f"Preprocessing dataset with shape {df.shape}")
        
        df = df.dropna()
        
        if target_col is not None:
            if target_col not in df.columns:
                logger.error(f"Target column '{target_col}' not found in dataset")
                return None
                
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            X = pd.get_dummies(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            if scale:
                scaler = StandardScaler()
                X_train = pd.DataFrame(
                    scaler.fit_transform(X_train), 
                    columns=X_train.columns, 
                    index=X_train.index
                )
                X_test = pd.DataFrame(
                    scaler.transform(X_test), 
                    columns=X_test.columns, 
                    index=X_test.index
                )
                
            logger.info(f"Preprocessing complete: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
            return X_train, X_test, y_train, y_test
        else:
            df = pd.get_dummies(df)
            
            if scale:
                scaler = StandardScaler()
                df = pd.DataFrame(
                    scaler.fit_transform(df), 
                    columns=df.columns, 
                    index=df.index
                )
                
            logger.info(f"Preprocessing complete: df shape {df.shape}")
            return df, None, None, None
            
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        return None


class ObjectStoreDataManager:
    def __init__(self):
        self.file_registry = {}  
        logger.info("ObjectStoreDataManager initialized")
    
    def store_file_data(self, filename, records):
        try:
            import pandas as pd
            
            df = pd.DataFrame(records)
            
            data_ref = ray.put(df)
            
            metadata = {
                'rows': len(df),
                'columns': list(df.columns),
                'loaded_at': pd.Timestamp.now().isoformat(),
                'data_ref': data_ref
            }
            
            self.file_registry[filename] = metadata
            
            logger.info(f"File {filename} stored in object store: {metadata['rows']} rows, {len(metadata['columns'])} columns")
            return True
        except Exception as e:
            logger.error(f"Error storing file {filename} in object store: {e}")
            return False
    
    def get_file_data(self, filename):
        try:
            if filename not in self.file_registry:
                return None
            
            data_ref = self.file_registry[filename]['data_ref']
            df = ray.get(data_ref)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error retrieving data for {filename} from object store: {e}")
            return None
    
    def get_file_info(self, filename):
        if filename in self.file_registry:
            info = self.file_registry[filename].copy()
            info.pop('data_ref', None)
            return info
        return None
    
    def get_file_columns(self, filename):
        info = self.get_file_info(filename)
        return info.get('columns', []) if info else []
    
    def get_file_sample(self, filename, n=5):
        try:
            if filename not in self.file_registry:
                return None
            
            data_ref = self.file_registry[filename]['data_ref']
            df = ray.get(data_ref)
            return df.head(n).to_dict('records')
        except Exception as e:
            logger.error(f"Error retrieving sample for {filename} from object store: {e}")
            return None
    
    def list_files(self):
        return list(self.file_registry.keys())
    
    def remove_file(self, filename):
        try:
            if filename in self.file_registry:
                del self.file_registry[filename]
                logger.info(f"File {filename} removed from object store")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing file {filename}: {e}")
            return False
    
_data_manager = None

def get_data_manager():
    global _data_manager
    if _data_manager is None:
        _data_manager = ObjectStoreDataManager()
    return _data_manager
