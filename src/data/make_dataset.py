import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

class DatasetProcessor:
    def __init__(self, dataset_name, test_size=0.9, random_state=42):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self.raw_dir = RAW_DATA_DIR / dataset_name
        self.processed_dir = PROCESSED_DATA_DIR / dataset_name
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        
    def load_data(self):
        raise NotImplementedError
        
    def preprocess(self, df):
        raise NotImplementedError
        
    def split_and_save(self, X, y):
        print(f"Creating {100*(1-self.test_size):.0f}% Train / {100*self.test_size:.0f}% Test split stratifying by labels...")
        
        # Train-test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
        
        # Convert to float32 for PyTorch memory efficiency
        np.save(self.processed_dir / "X_train.npy", X_train.astype(np.float32))
        np.save(self.processed_dir / "X_test.npy", X_test.astype(np.float32))
        np.save(self.processed_dir / "y_train.npy", y_train.astype(np.int64))
        np.save(self.processed_dir / "y_test.npy", y_test.astype(np.int64))
        
        print(f"Saved processed Numpy arrays to {self.processed_dir}")

class NSLKDDPorcessor(DatasetProcessor):
    def __init__(self):
        super().__init__("nsl-kdd")
        self.columns = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
            "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
        ]
        self.categorical_cols = ["protocol_type", "service", "flag"]
        
    def load_data(self):
        train_path = self.raw_dir / "KDDTrain+.txt"
        test_path = self.raw_dir / "KDDTest+.txt"
        
        df_train = pd.read_csv(train_path, header=None, names=self.columns)
        df_test = pd.read_csv(test_path, header=None, names=self.columns)
        df = pd.concat([df_train, df_test], ignore_index=True)
        return df
        
    def preprocess(self, df):
        print(f"Original NSL-KDD shape: {df.shape}")
        # Drop difficulty as it's not a real feature
        df = df.drop(columns=["difficulty"], errors='ignore')
        
        # 1. One-hot encode categorical features Let's use get_dummies
        df = pd.get_dummies(df, columns=self.categorical_cols)
        
        # 2. Extract features and labels
        y = np.where(df["label"] == "normal", 0, 1)  # 0: normal, 1: attack
        X = df.drop("label", axis=1).values
        
        # 3. Scale numerical features
        X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        
        print(f"Processed NSL-KDD (Features={X.shape[1]})")
        return X, y

class EdgeIIoTProcessor(DatasetProcessor):
    def __init__(self):
        super().__init__("edge-iiotset")
        
    def load_data(self):
        path = self.raw_dir / "ML-EdgeIIoT-dataset.csv"
        df = pd.read_csv(path, low_memory=False)
        return df
        
    def preprocess(self, df):
        print(f"Original Edge-IIoTset shape: {df.shape}")
        
        # Check column names since sometimes there are leading/trailing spaces
        df.columns = df.columns.str.strip()

        # Drop columns with high NaN ratio or irrelevant ID info
        drop_cols = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", 
                     "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
                     "icmp.transmit_timestamp", "icmp.unused", "http.tls_port",
                     "dns.qry.name.len", "mqtt.msg", "mqtt.conack.val"]
        
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
        
        # Fill NaNs with 0 (safe default for missing network metric flags)
        df = df.fillna(0)
        
        # Label encode remaining object columns
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ["Attack_label", "Attack_type"]:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                except ValueError:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                    
        # Extract features and labels
        if "Attack_label" in df.columns:
            y = df["Attack_label"].values  # 0 for normal, 1 for attack
            X = df.drop(columns=["Attack_label", "Attack_type"], errors='ignore').values
        else:
            raise KeyError("Attack_label column not found in Edge-IIoTset")
            
        # Scale numerical features
        X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        
        print(f"Processed Edge-IIoTset (Features={X.shape[1]})")
        return X, y

class CTU13Processor(DatasetProcessor):
    def __init__(self):
        super().__init__("ctu-13")
        
    def load_data(self):
        attack_path = self.raw_dir / "CTU13_Attack_Traffic.csv"
        normal_path = self.raw_dir / "CTU13_Normal_Traffic.csv"
        
        df_attack = pd.read_csv(attack_path)
        df_normal = pd.read_csv(normal_path)
        df = pd.concat([df_attack, df_normal], ignore_index=True)
        return df

    def preprocess(self, df):
        print(f"Original CTU-13 shape: {df.shape}")
        # Drop irrevelant columns like 'Unnamed: 0' if exists
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")
        
        # CTU-13 usually has 'Label' or similar
        label_col = "Label" if "Label" in df.columns else df.columns[-1]
        
        df = df.fillna(0)
        
        # Label encode strings
        for col in df.columns:
            if df[col].dtype == 'object' and col != label_col:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                
        # Binarize label
        if df[label_col].dtype == 'object':
            # E.g. 'Normal' vs 'Botnet'
            y = np.where(df[label_col].astype(str).str.lower().str.contains("normal"), 0, 1)
        else:
            y = df[label_col].values
            
        X = df.drop(columns=[label_col]).values
        X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        
        print(f"Processed CTU-13 (Features={X.shape[1]})")
        return X, y

class UKMIDS20Processor(DatasetProcessor):
    def __init__(self):
        super().__init__("ukm-ids20")
        
    def load_data(self):
        train_path = self.raw_dir / "UKM-IDS20 Training set.csv"
        test_path = self.raw_dir / "UKM-IDS20 Testing set.csv"
        
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df = pd.concat([df_train, df_test], ignore_index=True)
        return df

    def preprocess(self, df):
        print(f"Original UKM-IDS20 shape: {df.shape}")
        df = df.fillna(0)
        
        # Known label column for UKM-IDS20
        label_col = "Class name" if "Class name" in df.columns else df.columns[-1]
        
        # Label encode objects
        for col in df.columns:
            if df[col].dtype == 'object' and col != label_col:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                
        # Binarize label
        if df[label_col].dtype == 'object':
            y = np.where(df[label_col].astype(str).str.lower().str.contains("normal"), 0, 1)
        else:
            y = df[label_col].values
            
        X = df.drop(columns=[label_col]).values
        X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        
        print(f"Processed UKM-IDS20 (Features={X.shape[1]})")
        return X, y

def main():
    processors = {
        "nsl-kdd": NSLKDDPorcessor(),
        "edge-iiotset": EdgeIIoTProcessor(),
        "ctu-13": CTU13Processor(),
        "ukm-ids20": UKMIDS20Processor()
    }
    
    for name, processor in processors.items():
        try:
            print(f"\\n{'='*50}\\nProcessing {name}...\\n{'='*50}")
            df = processor.load_data()
            X, y = processor.preprocess(df)
            processor.split_and_save(X, y)
        except Exception as e:
            print(f"Error processing {name}: {e}")

if __name__ == "__main__":
    main()
