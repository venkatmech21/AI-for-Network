#!/usr/bin/env python3
"""
Real-time Network Anomaly Detection Pipeline
Fixed version with proper detector initialization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import joblib
import os
import sys

# Add the main detector class (simplified version for pipeline)
class SimpleAnomalyDetector:
    """
    Simplified anomaly detector for real-time monitoring
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
    def extract_syslog_features(self, syslog_df):
        """
        Extract basic features from syslog data for real-time processing
        """
        features_df = syslog_df.copy()
        
        # Basic temporal features
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Log level encoding (simple mapping)
        level_map = {'INFO': 0, 'WARNING': 1, 'ERROR': 2, 'CRITICAL': 3}
        features_df['level_encoded'] = features_df['level'].map(level_map).fillna(0)
        
        # Message features
        features_df['message_length'] = features_df['message'].str.len()
        features_df['word_count'] = features_df['message'].str.split().str.len()
        features_df['uppercase_ratio'] = features_df['message'].str.count(r'[A-Z]') / features_df['message_length']
        features_df['digit_ratio'] = features_df['message'].str.count(r'\d') / features_df['message_length']
        features_df['special_char_ratio'] = features_df['message'].str.count(r'[^a-zA-Z0-9\s]') / features_df['message_length']
        
        # IP address features
        import re
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        features_df['ip_count'] = features_df['message'].str.findall(ip_pattern).str.len()
        features_df['has_ip'] = (features_df['ip_count'] > 0).astype(int)
        
        # Error keywords
        error_keywords = ['error', 'fail', 'critical', 'alert', 'warning', 'exception', 'timeout']
        for keyword in error_keywords:
            features_df[f'has_{keyword}'] = features_df['message'].str.lower().str.contains(keyword).astype(int)
        
        # Select final features
        feature_columns = [
            'hour', 'day_of_week', 'is_weekend', 'level_encoded',
            'message_length', 'word_count', 'uppercase_ratio', 
            'digit_ratio', 'special_char_ratio', 'ip_count', 'has_ip'
        ] + [f'has_{kw}' for kw in error_keywords]
        
        return features_df[feature_columns].fillna(0)
    
    def extract_flow_features(self, flow_df):
        """
        Extract basic features from flow data for real-time processing
        """
        features_df = flow_df.copy()
        
        # Basic flow features
        features_df['packets_per_second'] = features_df['packets'] / features_df['duration']
        features_df['bytes_per_second'] = features_df['bytes'] / features_df['duration']
        features_df['bytes_per_packet'] = features_df['bytes'] / features_df['packets']
        
        # Protocol encoding
        protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
        features_df['protocol_encoded'] = features_df['protocol'].map(protocol_map).fillna(0)
        
        # Port features
        features_df['is_well_known_src'] = (features_df['src_port'] < 1024).astype(int)
        features_df['is_well_known_dst'] = (features_df['dst_port'] < 1024).astype(int)
        features_df['is_common_port'] = features_df['dst_port'].isin([22, 23, 53, 80, 443, 993, 995]).astype(int)
        
        # Time features
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['is_business_hours'] = features_df['hour'].between(9, 17).astype(int)
        
        # IP class features (simplified)
        features_df['src_ip_class'] = features_df['src_ip'].str.split('.').str[0].astype(int)
        features_df['dst_ip_class'] = features_df['dst_ip'].str.split('.').str[0].astype(int)
        features_df['is_internal_src'] = features_df['src_ip_class'].isin([10, 172, 192]).astype(int)
        features_df['is_internal_dst'] = features_df['dst_ip_class'].isin([10, 172, 192]).astype(int)
        
        # Statistical features
        features_df['duration_log'] = np.log1p(features_df['duration'])
        features_df['packets_log'] = np.log1p(features_df['packets'])
        features_df['bytes_log'] = np.log1p(features_df['bytes'])
        
        # Select final features
        feature_columns = [
            'src_port', 'dst_port', 'protocol_encoded', 'duration', 'packets', 'bytes',
            'packets_per_second', 'bytes_per_second', 'bytes_per_packet',
            'is_well_known_src', 'is_well_known_dst', 'is_common_port',
            'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            'src_ip_class', 'dst_ip_class', 'is_internal_src', 'is_internal_dst',
            'duration_log', 'packets_log', 'bytes_log'
        ]
        
        final_features = features_df[feature_columns].fillna(0)
        
        # Handle infinite values
        final_features = final_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return final_features
    
    def create_simple_models(self):
        """
        Create simple anomaly detection models for demonstration
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        
        # Create simple models with default parameters
        self.models = {
            'syslog': {
                'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
                'one_class_svm': OneClassSVM(nu=0.1)
            },
            'flow': {
                'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
                'one_class_svm': OneClassSVM(nu=0.1)
            }
        }
        
        self.scalers = {
            'syslog': StandardScaler(),
            'flow': StandardScaler()
        }
        
        print("⚠️  Using simple default models for demonstration.")
        print("   For production, train models on your actual data first.")
        
    def train_on_sample_data(self):
        """
        Train models on small sample data for demonstration
        """
        print("🔄 Training on sample data...")
        
        # Generate small sample datasets
        sample_syslog = []
        for i in range(100):
            sample_syslog.append({
                'timestamp': datetime.now() - timedelta(minutes=i),
                'level': np.random.choice(['INFO', 'WARNING', 'ERROR']),
                'message': f"Sample log message {i} with some content",
                'source': f"server{i % 5}"
            })
        
        sample_flow = []
        for i in range(100):
            sample_flow.append({
                'src_ip': f"192.168.1.{i % 50 + 1}",
                'dst_ip': f"10.0.0.{i % 20 + 1}",
                'src_port': np.random.randint(1024, 65535),
                'dst_port': np.random.choice([80, 443, 22, 53]),
                'protocol': np.random.choice(['TCP', 'UDP']),
                'duration': np.random.exponential(30),
                'packets': np.random.poisson(50),
                'bytes': np.random.normal(1000, 300),
                'timestamp': datetime.now() - timedelta(minutes=i)
            })
        
        # Extract features
        syslog_features = self.extract_syslog_features(pd.DataFrame(sample_syslog))
        flow_features = self.extract_flow_features(pd.DataFrame(sample_flow))
        
        # Scale and train
        syslog_scaled = self.scalers['syslog'].fit_transform(syslog_features)
        flow_scaled = self.scalers['flow'].fit_transform(flow_features)
        
        # Train models
        self.models['syslog']['isolation_forest'].fit(syslog_scaled)
        self.models['syslog']['one_class_svm'].fit(syslog_scaled)
        self.models['flow']['isolation_forest'].fit(flow_scaled)
        self.models['flow']['one_class_svm'].fit(flow_scaled)
        
        self.is_trained = True
        print("✅ Sample training completed.")
    
    def predict_anomaly(self, features, data_type):
        """
        Predict if data is anomalous
        """
        if not self.is_trained:
            return {'error': 'Models not trained'}
        
        # Scale features
        features_scaled = self.scalers[data_type].transform(features)
        
        predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models[data_type].items():
            try:
                pred = model.predict(features_scaled)
                predictions[model_name] = (pred == -1).astype(int)[0]  # 1 if anomaly, 0 if normal
            except Exception as e:
                predictions[model_name] = 0  # Default to normal if error
        
        # Ensemble prediction (majority vote)
        ensemble_pred = int(sum(predictions.values()) >= len(predictions) / 2)
        confidence = sum(predictions.values()) / len(predictions)
        
        return {
            'is_anomaly': bool(ensemble_pred),
            'confidence': float(confidence),
            'individual_models': predictions
        }


# Global detector instance
detector = SimpleAnomalyDetector()


def initialize_detector():
    """
    Initialize the detector with trained models
    """
    global detector
    
    # Try to load pre-trained models if they exist
    model_file = 'trained_detector.pkl'
    
    if os.path.exists(model_file):
        try:
            print(f"📂 Loading pre-trained models from {model_file}...")
            detector = joblib.load(model_file)
            print("✅ Pre-trained models loaded successfully!")
            return True
        except Exception as e:
            print(f"⚠️  Error loading models: {e}")
    
    # If no pre-trained models, create and train simple ones
    print("🔧 Creating and training simple demonstration models...")
    detector.create_simple_models()
    detector.train_on_sample_data()
    
    # Save the trained detector
    try:
        joblib.dump(detector, model_file)
        print(f"💾 Detector saved to {model_file}")
    except Exception as e:
        print(f"⚠️  Could not save detector: {e}")
    
    return True


def real_time_monitoring_pipeline(new_syslog_entry=None, new_flow_data=None):
    """
    Real-time anomaly detection pipeline
    """
    global detector
    
    if not detector.is_trained:
        print("⚠️  Detector not initialized. Call initialize_detector() first.")
        return {'error': 'Detector not trained'}
    
    results = {}
    
    if new_syslog_entry:
        try:
            # Extract features from new syslog entry
            syslog_df = pd.DataFrame([new_syslog_entry])
            syslog_features = detector.extract_syslog_features(syslog_df)
            
            # Detect anomalies
            results['syslog'] = detector.predict_anomaly(syslog_features, 'syslog')
            
        except Exception as e:
            results['syslog'] = {'error': f'Syslog processing error: {e}'}
    
    if new_flow_data:
        try:
            # Extract features from new flow data
            flow_df = pd.DataFrame([new_flow_data])
            flow_features = detector.extract_flow_features(flow_df)
            
            # Detect anomalies
            results['flow'] = detector.predict_anomaly(flow_features, 'flow')
            
        except Exception as e:
            results['flow'] = {'error': f'Flow processing error: {e}'}
    
    return results


def monitor_batch_data(syslog_entries=None, flow_entries=None):
    """
    Monitor a batch of data entries
    """
    results = []
    
    if syslog_entries:
        for entry in syslog_entries:
            result = real_time_monitoring_pipeline(new_syslog_entry=entry)
            result['timestamp'] = entry.get('timestamp', datetime.now())
            result['data_type'] = 'syslog'
            result['entry'] = entry
            results.append(result)
    
    if flow_entries:
        for entry in flow_entries:
            result = real_time_monitoring_pipeline(new_flow_data=entry)
            result['timestamp'] = entry.get('timestamp', datetime.now())
            result['data_type'] = 'flow'
            result['entry'] = entry
            results.append(result)
    
    return results


def generate_alert(result, threshold=0.5):
    """
    Generate alert if anomaly confidence exceeds threshold
    """
    if 'syslog' in result and result['syslog'].get('is_anomaly') and result['syslog'].get('confidence', 0) > threshold:
        return {
            'alert_type': 'SYSLOG_ANOMALY',
            'severity': 'HIGH' if result['syslog']['confidence'] > 0.8 else 'MEDIUM',
            'confidence': result['syslog']['confidence'],
            'timestamp': datetime.now(),
            'details': result['syslog']
        }
    
    if 'flow' in result and result['flow'].get('is_anomaly') and result['flow'].get('confidence', 0) > threshold:
        return {
            'alert_type': 'FLOW_ANOMALY',
            'severity': 'HIGH' if result['flow']['confidence'] > 0.8 else 'MEDIUM',
            'confidence': result['flow']['confidence'],
            'timestamp': datetime.now(),
            'details': result['flow']
        }
    
    return None


def main_demo():
    """
    Demonstration of the monitoring pipeline
    """
    print("🚀 Network Anomaly Detection Pipeline - Demo")
    print("=" * 50)
    
    # Initialize detector
    if not initialize_detector():
        print("❌ Failed to initialize detector")
        return
    
    print("\n🔍 Testing real-time monitoring...")
    
    # Sample data for testing
    sample_syslog_normal = {
        'timestamp': datetime.now(),
        'level': 'INFO',
        'message': 'User john logged in successfully from 192.168.1.100',
        'source': 'auth-server'
    }
    
    sample_syslog_suspicious = {
        'timestamp': datetime.now(),
        'level': 'ERROR',
        'message': 'CRITICAL: Multiple failed login attempts detected from 192.168.1.100',
        'source': 'auth-server'
    }
    
    sample_flow_normal = {
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.5',
        'src_port': 54321,
        'dst_port': 443,
        'protocol': 'TCP',
        'duration': 30.5,
        'packets': 45,
        'bytes': 50000,
        'timestamp': datetime.now()
    }
    
    sample_flow_suspicious = {
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.5',
        'src_port': 54321,
        'dst_port': 22,
        'protocol': 'TCP',
        'duration': 0.1,
        'packets': 1,
        'bytes': 64,
        'timestamp': datetime.now()
    }
    
    # Test normal data
    print("\n📊 Testing normal data:")
    normal_result = real_time_monitoring_pipeline(sample_syslog_normal, sample_flow_normal)
    print(f"Normal Syslog: {normal_result.get('syslog', {})}")
    print(f"Normal Flow: {normal_result.get('flow', {})}")
    
    # Test suspicious data
    print("\n🚨 Testing suspicious data:")
    suspicious_result = real_time_monitoring_pipeline(sample_syslog_suspicious, sample_flow_suspicious)
    print(f"Suspicious Syslog: {suspicious_result.get('syslog', {})}")
    print(f"Suspicious Flow: {suspicious_result.get('flow', {})}")
    
    # Generate alerts
    print("\n🔔 Alert generation:")
    alert1 = generate_alert(normal_result, threshold=0.5)
    alert2 = generate_alert(suspicious_result, threshold=0.5)
    
    if alert1:
        print(f"Normal data alert: {alert1}")
    else:
        print("No alert for normal data ✅")
    
    if alert2:
        print(f"Suspicious data alert: {alert2}")
    else:
        print("No alert for suspicious data")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    # Run demonstration
    main_demo()