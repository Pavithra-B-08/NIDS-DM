import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Column names for KDD Cup 99 dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

def load_and_preprocess_data(train_file, test_file):
    """Load and preprocess KDD dataset"""
    print("Loading datasets...")
    
    # Load training data - specify label column as string
    train_data = pd.read_csv(train_file, names=column_names, dtype={'label': str})
    test_data = pd.read_csv(test_file, names=column_names, dtype={'label': str})
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Convert label to string and remove dots
    train_data['label'] = train_data['label'].astype(str).str.replace('.', '', regex=False)
    test_data['label'] = test_data['label'].astype(str).str.replace('.', '', regex=False)
    
    # Categorize attacks into broader classes
    attack_mapping = {
        'normal': 'normal',
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos',
        'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos', 'processtable': 'dos',
        'udpstorm': 'dos',
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
        'mscan': 'probe', 'saint': 'probe',
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
        'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
        'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l',
        'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
        'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
    }
    
    train_data['attack_category'] = train_data['label'].map(attack_mapping)
    test_data['attack_category'] = test_data['label'].map(attack_mapping)
    
    # Handle unknown attack types
    train_data['attack_category'].fillna('unknown', inplace=True)
    test_data['attack_category'].fillna('unknown', inplace=True)
    
    # Print class distribution
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION IN TRAINING DATA:")
    print("="*60)
    print(train_data['attack_category'].value_counts())
    print(f"\nTotal samples: {len(train_data)}")
    print(f"Normal traffic: {len(train_data[train_data['attack_category'] == 'normal'])} ({len(train_data[train_data['attack_category'] == 'normal'])/len(train_data)*100:.2f}%)")
    print("="*60)
    
    return train_data, test_data

def feature_engineering(train_data, test_data):
    """Encode categorical features and scale numerical features"""
    print("\nPerforming feature engineering...")
    
    # Create copies to avoid modifying original data
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    categorical_columns = ['protocol_type', 'service', 'flag']
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        # Fit on combined data to handle unseen categories
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le
        
        print(f"  Encoded {col}: {len(le.classes_)} unique values")
    
    # Encode target labels
    target_encoder = LabelEncoder()
    train_df['target'] = target_encoder.fit_transform(train_df['attack_category'])
    test_df['target'] = target_encoder.transform(test_df['attack_category'])
    
    # Print encoding mapping
    print("\n" + "="*60)
    print("TARGET ENCODING MAPPING:")
    print("="*60)
    for class_name, encoded_value in zip(target_encoder.classes_, 
                                         target_encoder.transform(target_encoder.classes_)):
        print(f"  {class_name:15s} -> {encoded_value}")
    print("="*60)
    
    # Separate features and target
    feature_columns = [col for col in train_df.columns if col not in ['label', 'attack_category', 'target']]
    
    X_train = train_df[feature_columns].copy()
    y_train = train_df['target'].copy()
    X_test = test_df[feature_columns].copy()
    y_test = test_df['target'].copy()
    
    # Convert all features to numeric (in case any are still object type)
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"\nFeature columns: {len(feature_columns)}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, target_encoder, label_encoders, feature_columns

def train_model(X_train, y_train):
    """Train Random Forest classifier with class balancing"""
    print("\nTraining Random Forest model with class balancing...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',  # CRITICAL FIX: Balance classes
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    return model

def evaluate_model(model, X_test, y_test, target_encoder):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT:")
    print("="*60)
    print(classification_report(y_test, y_pred, 
                                target_names=target_encoder.classes_,
                                zero_division=0))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX:")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Print detailed confusion matrix with labels
    print("\nDetailed Confusion Matrix:")
    cm_df = pd.DataFrame(cm, 
                         index=target_encoder.classes_, 
                         columns=target_encoder.classes_)
    print(cm_df)
    
    # Calculate per-class metrics
    print("\n" + "="*60)
    print("PER-CLASS PERFORMANCE:")
    print("="*60)
    for i, class_name in enumerate(target_encoder.classes_):
        mask = y_test == i
        if mask.sum() > 0:
            class_accuracy = (y_pred[mask] == i).sum() / mask.sum()
            print(f"  {class_name:15s}: {class_accuracy*100:.2f}% ({mask.sum()} samples)")
    print("="*60)
    
    return accuracy

def save_models(model, scaler, target_encoder, label_encoders, feature_columns):
    """Save trained models and encoders"""
    print("\nSaving models...")
    
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/nids_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(target_encoder, 'models/target_encoder.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(feature_columns, 'models/feature_columns.pkl')
    
    print("Models saved successfully!")
    print("  - nids_model.pkl")
    print("  - scaler.pkl")
    print("  - target_encoder.pkl")
    print("  - label_encoders.pkl")
    print("  - feature_columns.pkl")

def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("NETWORK INTRUSION DETECTION SYSTEM - TRAINING PIPELINE")
    print("="*60)
    
    # File paths
    train_file = 'data/KDDTrain+.txt'
    test_file = 'data/KDDTest+.txt'
    
    # Load and preprocess data
    train_data, test_data = load_and_preprocess_data(train_file, test_file)
    
    # Feature engineering
    X_train, X_test, y_train, y_test, scaler, target_encoder, label_encoders, feature_columns = feature_engineering(train_data, test_data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, target_encoder)
    
    # Save models
    save_models(model, scaler, target_encoder, label_encoders, feature_columns)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now run: python app.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()