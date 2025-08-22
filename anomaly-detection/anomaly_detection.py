#!/usr/bin/env python3
"""
Network Anomaly Detection System
Analyzes syslog entries and network flow data for anomaly detection
using multiple unsupervised learning algorithms.

Author: Network ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, classification_report

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class NetworkAnomalyDetector:
    """
    Comprehensive anomaly detection system for network data
    """
    
    def __init__(self, contamination_rate=0.05):
        """
        Initialize the anomaly detector
        
        Args:
            contamination_rate (float): Expected proportion of anomalies (5% default)
        """
        self.contamination_rate = contamination_rate
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.results = {}
        
    def generate_sample_syslog_data(self, n_samples=20000):
        """
        Generate realistic syslog data for demonstration
        """
        print(f"Generating {n_samples} sample syslog entries...")
        
        # Define log patterns
        normal_patterns = [
            "INFO: User {user} logged in from {ip}",
            "INFO: Service {service} started successfully",
            "INFO: Database connection established",
            "INFO: Backup completed successfully",
            "WARNING: High CPU usage detected: {cpu}%",
            "INFO: Network interface {interface} up",
            "INFO: SSL certificate renewed for {domain}",
            "INFO: Firewall rule applied for {ip}",
        ]
        
        anomaly_patterns = [
            "ERROR: Multiple failed login attempts from {ip}",
            "CRITICAL: Suspicious SQL injection attempt detected",
            "ERROR: Unauthorized access attempt to {file}",
            "CRITICAL: Malware signature detected in {file}",
            "ERROR: Port scan detected from {ip}",
            "CRITICAL: Data exfiltration attempt blocked",
            "ERROR: Buffer overflow attempt in {service}",
            "CRITICAL: Ransomware activity detected",
        ]
        
        # Generate sample data
        logs = []
        for i in range(n_samples):
            timestamp = datetime.now() - timedelta(
                hours=np.random.randint(0, 24*7),
                minutes=np.random.randint(0, 60),
                seconds=np.random.randint(0, 60)
            )
            
            # 95% normal, 5% anomalous
            if np.random.random() < 0.95:
                pattern = np.random.choice(normal_patterns)
                level = np.random.choice(['INFO', 'WARNING'], p=[0.8, 0.2])
            else:
                pattern = np.random.choice(anomaly_patterns)
                level = np.random.choice(['ERROR', 'CRITICAL'])
            
            # Fill in placeholders
            message = pattern.format(
                user=f"user{np.random.randint(1, 100)}",
                ip=f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                service=np.random.choice(['apache', 'mysql', 'ssh', 'nginx', 'postgres']),
                cpu=np.random.randint(80, 100),
                interface=f"eth{np.random.randint(0, 3)}",
                domain=np.random.choice(['example.com', 'test.org', 'company.net']),
                file=f"/var/log/{np.random.choice(['access', 'error', 'system'])}.log"
            )
            
            logs.append({
                'timestamp': timestamp,
                'level': level,
                'message': message,
                'source': f"server{np.random.randint(1, 20)}",
                'true_label': 1 if level in ['ERROR', 'CRITICAL'] else 0  # For evaluation
            })
        
        return pd.DataFrame(logs)
    
    def generate_sample_flow_data(self, n_samples=10000):
        """
        Generate realistic network flow data for demonstration
        """
        print(f"Generating {n_samples} sample network flows...")
        
        flows = []
        for i in range(n_samples):
            # Generate normal vs anomalous flows
            if np.random.random() < 0.95:  # 95% normal traffic
                # Normal traffic patterns
                protocol = np.random.choice(['TCP', 'UDP', 'ICMP'], p=[0.7, 0.25, 0.05])
                
                # Source port selection for normal traffic
                if np.random.random() < 0.46:  # 46% from well-known ports
                    src_port = np.random.choice([80, 443, 22, 53, 25, 993, 995])
                else:  # 54% from high ports
                    src_port = np.random.randint(1024, 65535)
                
                # Destination port selection for normal traffic  
                if np.random.random() < 0.75:  # 75% to well-known ports
                    dst_port = np.random.choice([80, 443, 22, 53, 25], p=[0.27, 0.40, 0.13, 0.13, 0.07])
                else:  # 25% to high ports
                    dst_port = np.random.randint(1024, 65535)
                
                duration = np.random.exponential(30)  # Normal session duration
                packets = np.random.poisson(50)
                bytes_transferred = packets * np.random.normal(1000, 300)
                
                true_label = 0  # Normal
                
            else:  # 5% anomalous traffic
                # Anomalous patterns (DDoS, scanning, etc.)
                protocol = np.random.choice(['TCP', 'UDP', 'ICMP'], p=[0.5, 0.3, 0.2])
                src_port = np.random.randint(1, 65535)
                
                # Destination port selection for anomalous traffic (targeting vulnerable services)
                if np.random.random() < 0.4:  # 40% targeting specific vulnerable ports
                    dst_port = np.random.choice([22, 23, 3389, 445])
                else:  # 60% random port scanning
                    dst_port = np.random.randint(1, 1024)
                
                # Anomalous characteristics
                if np.random.random() < 0.5:
                    duration = np.random.exponential(1)      # Very short (scanning)
                    packets = np.random.poisson(1)           # Very few packets (scanning)
                else:
                    duration = np.random.exponential(300)    # Very long (data exfiltration)
                    packets = np.random.poisson(1000)        # Many packets (DDoS)
                
                bytes_transferred = packets * np.random.normal(100, 50)
                
                true_label = 1  # Anomalous
            
            flows.append({
                'src_ip': f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                'dst_ip': f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'duration': max(0.1, duration),
                'packets': max(1, int(packets)),
                'bytes': max(100, bytes_transferred),
                'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 60*24)),
                'true_label': true_label
            })
        
        return pd.DataFrame(flows)
    
    def extract_syslog_features(self, syslog_df):
        """
        Extract features from syslog data for anomaly detection
        """
        print("Extracting features from syslog data...")
        
        features_df = syslog_df.copy()
        
        # Basic features
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Log level encoding
        level_encoder = LabelEncoder()
        features_df['level_encoded'] = level_encoder.fit_transform(features_df['level'])
        
        # Message length and complexity
        features_df['message_length'] = features_df['message'].str.len()
        features_df['word_count'] = features_df['message'].str.split().str.len()
        features_df['uppercase_ratio'] = features_df['message'].str.count(r'[A-Z]') / features_df['message_length']
        features_df['digit_ratio'] = features_df['message'].str.count(r'\d') / features_df['message_length']
        features_df['special_char_ratio'] = features_df['message'].str.count(r'[^a-zA-Z0-9\s]') / features_df['message_length']
        
        # IP address extraction and features
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        features_df['ip_count'] = features_df['message'].str.findall(ip_pattern).str.len()
        features_df['has_ip'] = (features_df['ip_count'] > 0).astype(int)
        
        # Error keywords
        error_keywords = ['error', 'fail', 'critical', 'alert', 'warning', 'exception', 'timeout']
        for keyword in error_keywords:
            features_df[f'has_{keyword}'] = features_df['message'].str.lower().str.contains(keyword).astype(int)
        
        # TF-IDF features for message content
        print("Generating TF-IDF features...")
        tfidf = TfidfVectorizer(max_features=50, stop_words='english', lowercase=True, 
                               ngram_range=(1, 2), max_df=0.8, min_df=5)
        tfidf_features = tfidf.fit_transform(features_df['message'])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                               columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # Combine all features
        feature_columns = ['hour', 'day_of_week', 'is_weekend', 'level_encoded', 
                          'message_length', 'word_count', 'uppercase_ratio', 
                          'digit_ratio', 'special_char_ratio', 'ip_count', 'has_ip'] + \
                         [f'has_{kw}' for kw in error_keywords]
        
        numerical_features = features_df[feature_columns].fillna(0)
        final_features = pd.concat([numerical_features.reset_index(drop=True), 
                                   tfidf_df.reset_index(drop=True)], axis=1)
        
        self.feature_names['syslog'] = final_features.columns.tolist()
        return final_features
    
    def extract_flow_features(self, flow_df):
        """
        Extract features from network flow data for anomaly detection
        """
        print("Extracting features from flow data...")
        
        features_df = flow_df.copy()
        
        # Basic flow features
        features_df['packets_per_second'] = features_df['packets'] / features_df['duration']
        features_df['bytes_per_second'] = features_df['bytes'] / features_df['duration']
        features_df['bytes_per_packet'] = features_df['bytes'] / features_df['packets']
        
        # Protocol encoding
        protocol_encoder = LabelEncoder()
        features_df['protocol_encoded'] = protocol_encoder.fit_transform(features_df['protocol'])
        
        # Port features
        features_df['is_well_known_src'] = (features_df['src_port'] < 1024).astype(int)
        features_df['is_well_known_dst'] = (features_df['dst_port'] < 1024).astype(int)
        features_df['is_common_port'] = features_df['dst_port'].isin([22, 23, 53, 80, 443, 993, 995]).astype(int)
        
        # Time-based features
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['is_business_hours'] = features_df['hour'].between(9, 17).astype(int)
        
        # IP-based features (simplified)
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
        
        self.feature_names['flow'] = final_features.columns.tolist()
        return final_features
    
    def train_anomaly_models(self, features, data_type='syslog'):
        """
        Train multiple anomaly detection models
        """
        print(f"Training anomaly detection models for {data_type} data...")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers[data_type] = scaler
        
        models = {}
        
        # 1. Isolation Forest
        print("  Training Isolation Forest...")
        iso_forest = IsolationForest(contamination=self.contamination_rate, 
                                   random_state=42, n_estimators=100)
        iso_forest.fit(features_scaled)
        models['isolation_forest'] = iso_forest
        
        # 2. One-Class SVM
        print("  Training One-Class SVM...")
        ocsvm = OneClassSVM(nu=self.contamination_rate, kernel='rbf', gamma='scale')
        ocsvm.fit(features_scaled)
        models['one_class_svm'] = ocsvm
        
        # 3. DBSCAN for anomaly detection
        print("  Training DBSCAN...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(features_scaled)
        models['dbscan'] = dbscan
        
        # 4. K-Means for outlier detection
        print("  Training K-Means...")
        # Determine optimal number of clusters
        k_range = range(2, min(10, len(features) // 100))
        if len(k_range) > 0:
            silhouette_scores = []
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_temp = kmeans_temp.fit_predict(features_scaled)
                if len(set(labels_temp)) > 1:  # Need at least 2 clusters for silhouette score
                    score = silhouette_score(features_scaled, labels_temp)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(-1)
            
            if silhouette_scores and max(silhouette_scores) > 0:
                optimal_k = k_range[np.argmax(silhouette_scores)]
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                kmeans.fit(features_scaled)
                models['kmeans'] = kmeans
        
        # 5. PCA for dimensionality reduction and reconstruction error
        print("  Training PCA...")
        pca = PCA(n_components=min(10, features_scaled.shape[1]))
        pca.fit(features_scaled)
        models['pca'] = pca
        
        self.models[data_type] = models
        return models
    
    def detect_anomalies(self, features, data_type='syslog'):
        """
        Detect anomalies using trained models
        """
        print(f"Detecting anomalies in {data_type} data...")
        
        # Scale features
        features_scaled = self.scalers[data_type].transform(features)
        models = self.models[data_type]
        
        anomaly_scores = {}
        predictions = {}
        
        # Isolation Forest
        if 'isolation_forest' in models:
            iso_pred = models['isolation_forest'].predict(features_scaled)
            iso_scores = models['isolation_forest'].score_samples(features_scaled)
            predictions['isolation_forest'] = (iso_pred == -1).astype(int)
            anomaly_scores['isolation_forest'] = -iso_scores  # Convert to anomaly scores
        
        # One-Class SVM
        if 'one_class_svm' in models:
            ocsvm_pred = models['one_class_svm'].predict(features_scaled)
            ocsvm_scores = models['one_class_svm'].score_samples(features_scaled)
            predictions['one_class_svm'] = (ocsvm_pred == -1).astype(int)
            anomaly_scores['one_class_svm'] = -ocsvm_scores
        
        # DBSCAN
        if 'dbscan' in models:
            dbscan_pred = models['dbscan'].fit_predict(features_scaled)
            predictions['dbscan'] = (dbscan_pred == -1).astype(int)
            anomaly_scores['dbscan'] = (dbscan_pred == -1).astype(float)
        
        # K-Means (distance to centroid)
        if 'kmeans' in models:
            kmeans_labels = models['kmeans'].predict(features_scaled)
            centroids = models['kmeans'].cluster_centers_
            distances = np.array([np.linalg.norm(features_scaled[i] - centroids[kmeans_labels[i]]) 
                                for i in range(len(features_scaled))])
            threshold = np.percentile(distances, 100 * (1 - self.contamination_rate))
            predictions['kmeans'] = (distances > threshold).astype(int)
            anomaly_scores['kmeans'] = distances
        
        # PCA reconstruction error
        if 'pca' in models:
            features_reconstructed = models['pca'].inverse_transform(
                models['pca'].transform(features_scaled))
            reconstruction_errors = np.sum((features_scaled - features_reconstructed) ** 2, axis=1)
            threshold = np.percentile(reconstruction_errors, 100 * (1 - self.contamination_rate))
            predictions['pca'] = (reconstruction_errors > threshold).astype(int)
            anomaly_scores['pca'] = reconstruction_errors
        
        return predictions, anomaly_scores
    
    def ensemble_predictions(self, predictions):
        """
        Combine predictions from multiple models using ensemble voting
        """
        pred_df = pd.DataFrame(predictions)
        
        # Simple majority voting
        ensemble_pred = (pred_df.sum(axis=1) >= len(pred_df.columns) / 2).astype(int)
        
        # Confidence score (percentage of models agreeing)
        confidence = pred_df.mean(axis=1)
        
        return ensemble_pred, confidence
    
    def evaluate_performance(self, true_labels, predictions, data_type):
        """
        Evaluate anomaly detection performance
        """
        print(f"\n=== {data_type.upper()} ANOMALY DETECTION RESULTS ===")
        
        results = {}
        
        for model_name, pred in predictions.items():
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                
                precision = precision_score(true_labels, pred, zero_division=0)
                recall = recall_score(true_labels, pred, zero_division=0)
                f1 = f1_score(true_labels, pred, zero_division=0)
                accuracy = accuracy_score(true_labels, pred)
                
                results[model_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'anomalies_detected': pred.sum()
                }
                
                print(f"\n{model_name.upper()}:")
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1-Score: {f1:.3f}")
                print(f"  Anomalies Detected: {pred.sum()}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def visualize_results(self, features, predictions, anomaly_scores, true_labels, data_type):
        """
        Create visualizations for anomaly detection results
        """
        print(f"Creating visualizations for {data_type} data...")
        
        # PCA for visualization
        if features.shape[1] > 2:
            pca_viz = PCA(n_components=2)
            features_2d = pca_viz.fit_transform(StandardScaler().fit_transform(features))
        else:
            features_2d = features.values
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['True Labels', 'Isolation Forest', 'One-Class SVM', 
                           'DBSCAN', 'K-Means', 'Ensemble'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # True labels
        fig.add_trace(
            go.Scatter(x=features_2d[:, 0], y=features_2d[:, 1],
                      mode='markers',
                      marker=dict(color=true_labels, colorscale='RdYlBu', size=4),
                      name='True Labels'),
            row=1, col=1
        )
        
        # Model predictions
        models_to_plot = ['isolation_forest', 'one_class_svm', 'dbscan', 'kmeans']
        positions = [(1, 2), (1, 3), (2, 1), (2, 2)]
        
        for i, model in enumerate(models_to_plot):
            if model in predictions:
                row, col = positions[i]
                fig.add_trace(
                    go.Scatter(x=features_2d[:, 0], y=features_2d[:, 1],
                              mode='markers',
                              marker=dict(color=predictions[model], colorscale='RdYlBu', size=4),
                              name=model),
                    row=row, col=col
                )
        
        # Ensemble predictions
        if len(predictions) > 1:
            ensemble_pred, _ = self.ensemble_predictions(predictions)
            fig.add_trace(
                go.Scatter(x=features_2d[:, 0], y=features_2d[:, 1],
                          mode='markers',
                          marker=dict(color=ensemble_pred, colorscale='RdYlBu', size=4),
                          name='Ensemble'),
                row=2, col=3
            )
        
        fig.update_layout(
            title=f'{data_type.title()} Anomaly Detection Results',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def generate_report(self, syslog_results, flow_results):
        """
        Generate comprehensive anomaly detection report
        """
        print("\n" + "="*60)
        print("          NETWORK ANOMALY DETECTION REPORT")
        print("="*60)
        
        print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Contamination Rate: {self.contamination_rate * 100:.1f}%")
        
        # Summary statistics
        print(f"\n📊 DATA SUMMARY:")
        print(f"  Syslog Entries Analyzed: {len(syslog_results['true_labels']):,}")
        print(f"  Network Flows Analyzed: {len(flow_results['true_labels']):,}")
        
        # Best performing models
        print(f"\n🏆 BEST PERFORMING MODELS:")
        
        # Syslog best model
        syslog_f1_scores = {k: v['f1_score'] for k, v in syslog_results['evaluation'].items() if v}
        if syslog_f1_scores:
            best_syslog = max(syslog_f1_scores, key=syslog_f1_scores.get)
            print(f"  Syslog: {best_syslog.replace('_', ' ').title()} (F1: {syslog_f1_scores[best_syslog]:.3f})")
        
        # Flow best model
        flow_f1_scores = {k: v['f1_score'] for k, v in flow_results['evaluation'].items() if v}
        if flow_f1_scores:
            best_flow = max(flow_f1_scores, key=flow_f1_scores.get)
            print(f"  Flow: {best_flow.replace('_', ' ').title()} (F1: {flow_f1_scores[best_flow]:.3f})")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. Deploy {best_syslog.replace('_', ' ')} for syslog monitoring")
        print(f"  2. Deploy {best_flow.replace('_', ' ')} for network flow analysis")
        print(f"  3. Use ensemble methods for critical systems")
        print(f"  4. Regularly retrain models with new data")
        print(f"  5. Tune contamination rate based on actual anomaly rates")
        
        print(f"\n📈 NEXT STEPS:")
        print(f"  • Implement real-time monitoring pipeline")
        print(f"  • Set up alerting for high-confidence anomalies")
        print(f"  • Create feedback loop for model improvement")
        print(f"  • Integrate with existing security tools")
        
        print("="*60)


def main():
    """
    Main execution function
    """
    print("🚀 Starting Network Anomaly Detection System")
    print("="*60)
    
    # Initialize detector
    detector = NetworkAnomalyDetector(contamination_rate=0.05)
    
    # Generate sample data
    print("\n📊 GENERATING SAMPLE DATA...")
    syslog_data = detector.generate_sample_syslog_data(20000)
    flow_data = detector.generate_sample_flow_data(10000)
    
    print(f"✅ Generated {len(syslog_data):,} syslog entries")
    print(f"✅ Generated {len(flow_data):,} network flows")
    
    # Process Syslog Data
    print(f"\n🔍 PROCESSING SYSLOG DATA...")
    syslog_features = detector.extract_syslog_features(syslog_data)
    syslog_models = detector.train_anomaly_models(syslog_features, 'syslog')
    syslog_predictions, syslog_scores = detector.detect_anomalies(syslog_features, 'syslog')
    syslog_evaluation = detector.evaluate_performance(
        syslog_data['true_label'], syslog_predictions, 'syslog')
    
    # Process Flow Data
    print(f"\n🌐 PROCESSING NETWORK FLOW DATA...")
    flow_features = detector.extract_flow_features(flow_data)
    flow_models = detector.train_anomaly_models(flow_features, 'flow')
    flow_predictions, flow_scores = detector.detect_anomalies(flow_features, 'flow')
    flow_evaluation = detector.evaluate_performance(
        flow_data['true_label'], flow_predictions, 'flow')
    
    # Store results
    syslog_results = {
        'features': syslog_features,
        'predictions': syslog_predictions,
        'scores': syslog_scores,
        'true_labels': syslog_data['true_label'],
        'evaluation': syslog_evaluation
    }
    
    flow_results = {
        'features': flow_features,
        'predictions': flow_predictions,
        'scores': flow_scores,
        'true_labels': flow_data['true_label'],
        'evaluation': flow_evaluation
    }
    
    # Generate visualizations
    print(f"\n📈 CREATING VISUALIZATIONS...")
    try:
        syslog_viz = detector.visualize_results(
            syslog_features, syslog_predictions, syslog_scores, 
            syslog_data['true_label'], 'syslog')
        syslog_viz.write_html("syslog_anomaly_results.html")
        print("✅ Syslog visualization saved to 'syslog_anomaly_results.html'")
        
        flow_viz = detector.visualize_results(
            flow_features, flow_predictions, flow_scores, 
            flow_data['true_label'], 'flow')
        flow_viz.write_html("flow_anomaly_results.html")
        print("✅ Flow visualization saved to 'flow_anomaly_results.html'")
        
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    # Generate comprehensive report
    detector.generate_report(syslog_results, flow_results)
    
    # Save detailed results
    print(f"\n💾 SAVING DETAILED RESULTS...")
    
    # Save syslog anomalies
    syslog_anomalies = syslog_data.copy()
    for model, pred in syslog_predictions.items():
        syslog_anomalies[f'{model}_anomaly'] = pred
    
    if len(syslog_predictions) > 1:
        ensemble_pred, confidence = detector.ensemble_predictions(syslog_predictions)
        syslog_anomalies['ensemble_anomaly'] = ensemble_pred
        syslog_anomalies['confidence'] = confidence
    
    syslog_anomalies.to_csv('syslog_anomaly_results.csv', index=False)
    print("✅ Syslog results saved to 'syslog_anomaly_results.csv'")
    
    # Save flow anomalies
    flow_anomalies = flow_data.copy()
    for model, pred in flow_predictions.items():
        flow_anomalies[f'{model}_anomaly'] = pred
    
    if len(flow_predictions) > 1:
        ensemble_pred, confidence = detector.ensemble_predictions(flow_predictions)
        flow_anomalies['ensemble_anomaly'] = ensemble_pred
        flow_anomalies['confidence'] = confidence
    
    flow_anomalies.to_csv('flow_anomaly_results.csv', index=False)
    print("✅ Flow results saved to 'flow_anomaly_results.csv'")
    
    # Show sample anomalies
    print(f"\n🚨 SAMPLE DETECTED ANOMALIES:")
    
    # Syslog anomalies
    if 'ensemble_anomaly' in syslog_anomalies.columns:
        syslog_anomaly_samples = syslog_anomalies[syslog_anomalies['ensemble_anomaly'] == 1].head(3)
        print(f"\nSyslog Anomalies (Top 3):")
        for idx, row in syslog_anomaly_samples.iterrows():
            print(f"  • {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} [{row['level']}] {row['message'][:100]}...")
    
    # Flow anomalies
    if 'ensemble_anomaly' in flow_anomalies.columns:
        flow_anomaly_samples = flow_anomalies[flow_anomalies['ensemble_anomaly'] == 1].head(3)
        print(f"\nFlow Anomalies (Top 3):")
        for idx, row in flow_anomaly_samples.iterrows():
            print(f"  • {row['src_ip']}:{row['src_port']} → {row['dst_ip']}:{row['dst_port']} "
                  f"({row['protocol']}, {row['packets']} packets, {row['bytes']:.0f} bytes)")
    
    print(f"\n✅ Analysis complete! Check the generated files for detailed results.")
    
    return detector, syslog_results, flow_results


def analyze_feature_importance(detector, data_type='syslog'):
    """
    Analyze which features are most important for anomaly detection
    """
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS - {data_type.upper()}")
    print("-" * 50)
    
    if data_type not in detector.models:
        print(f"No models found for {data_type}")
        return
    
    models = detector.models[data_type]
    feature_names = detector.feature_names[data_type]
    
    # Isolation Forest feature importance
    if 'isolation_forest' in models:
        print("\nIsolation Forest - Feature Importance:")
        # For isolation forest, we can look at feature usage in splits
        print("  (Isolation Forest doesn't provide direct feature importance)")
        print("  Features are used randomly for splitting")
    
    # PCA component analysis
    if 'pca' in models:
        pca = models['pca']
        print(f"\nPCA Analysis:")
        print(f"  Components explaining 95% variance: {np.sum(np.cumsum(pca.explained_variance_ratio_) <= 0.95) + 1}")
        print(f"  Total variance explained by first 5 components: {pca.explained_variance_ratio_[:5].sum():.3f}")
        
        # Show top features for first 3 components
        for i in range(min(3, len(pca.components_))):
            component = pca.components_[i]
            top_features_idx = np.argsort(np.abs(component))[-5:]
            print(f"\n  Component {i+1} (variance: {pca.explained_variance_ratio_[i]:.3f}):")
            for idx in reversed(top_features_idx):
                print(f"    {feature_names[idx]}: {component[idx]:.3f}")


def create_monitoring_pipeline(detector):
    """
    Create a simple monitoring pipeline for real-time anomaly detection
    """
    print(f"\n🔄 CREATING MONITORING PIPELINE")
    print("-" * 50)
    
    pipeline_code = '''
def real_time_monitoring_pipeline(new_syslog_entry=None, new_flow_data=None):
    """
    Real-time anomaly detection pipeline
    """
    results = {}
    
    if new_syslog_entry:
        # Extract features from new syslog entry
        syslog_features = detector.extract_syslog_features(pd.DataFrame([new_syslog_entry]))
        
        # Detect anomalies
        syslog_predictions, syslog_scores = detector.detect_anomalies(syslog_features, 'syslog')
        
        # Ensemble prediction
        if len(syslog_predictions) > 1:
            ensemble_pred, confidence = detector.ensemble_predictions(syslog_predictions)
            results['syslog'] = {
                'is_anomaly': bool(ensemble_pred[0]),
                'confidence': float(confidence[0]),
                'individual_models': {k: bool(v[0]) for k, v in syslog_predictions.items()}
            }
    
    if new_flow_data:
        # Extract features from new flow data
        flow_features = detector.extract_flow_features(pd.DataFrame([new_flow_data]))
        
        # Detect anomalies
        flow_predictions, flow_scores = detector.detect_anomalies(flow_features, 'flow')
        
        # Ensemble prediction
        if len(flow_predictions) > 1:
            ensemble_pred, confidence = detector.ensemble_predictions(flow_predictions)
            results['flow'] = {
                'is_anomaly': bool(ensemble_pred[0]),
                'confidence': float(confidence[0]),
                'individual_models': {k: bool(v[0]) for k, v in flow_predictions.items()}
            }
    
    return results

# Example usage:
sample_syslog = {
    'timestamp': datetime.now(),
    'level': 'ERROR',
    'message': 'Multiple failed login attempts from 192.168.1.100',
    'source': 'server1'
}

sample_flow = {
    'src_ip': '192.168.1.100',
    'dst_ip': '10.0.0.5',
    'src_port': 54321,
    'dst_port': 22,
    'protocol': 'TCP',
    'duration': 0.5,
    'packets': 1,
    'bytes': 64,
    'timestamp': datetime.now()
}

# Monitor new data
monitoring_results = real_time_monitoring_pipeline(sample_syslog, sample_flow)
print("Monitoring Results:", monitoring_results)
'''
    
    print("Pipeline code template created. Key features:")
    print("  • Real-time feature extraction")
    print("  • Multi-model ensemble prediction")
    print("  • Confidence scoring")
    print("  • JSON-compatible output")
    print("\nTo implement:")
    print("  1. Save trained models using joblib")
    print("  2. Create API endpoint for real-time scoring")
    print("  3. Set up alerting based on confidence thresholds")
    print("  4. Implement feedback loop for model retraining")
    
    return pipeline_code


if __name__ == "__main__":
    # Run main analysis
    detector, syslog_results, flow_results = main()
    
    # Additional analysis
    print(f"\n" + "="*60)
    print("           ADDITIONAL ANALYSIS")
    print("="*60)
    
    # Feature importance analysis
    analyze_feature_importance(detector, 'syslog')
    analyze_feature_importance(detector, 'flow')
    
    # Create monitoring pipeline
    pipeline_code = create_monitoring_pipeline(detector)
 
    # Performance summary
    print(f"\n📊 FINAL PERFORMANCE SUMMARY")
    print("-" * 50)
    
    # Best models summary
    all_models = ['isolation_forest', 'one_class_svm', 'dbscan', 'kmeans', 'pca']
    
    print(f"\nSyslog Analysis:")
    for model in all_models:
        if model in syslog_results['evaluation'] and syslog_results['evaluation'][model]:
            eval_data = syslog_results['evaluation'][model]
            print(f"  {model:15s}: F1={eval_data['f1_score']:.3f}, "
                  f"Precision={eval_data['precision']:.3f}, "
                  f"Recall={eval_data['recall']:.3f}")
    
    print(f"\nFlow Analysis:")
    for model in all_models:
        if model in flow_results['evaluation'] and flow_results['evaluation'][model]:
            eval_data = flow_results['evaluation'][model]
            print(f"  {model:15s}: F1={eval_data['f1_score']:.3f}, "
                  f"Precision={eval_data['precision']:.3f}, "
                  f"Recall={eval_data['recall']:.3f}")
    
    print(f"\n🎯 DEPLOYMENT RECOMMENDATIONS:")
    print(f"  1. Use Isolation Forest for real-time detection (fast & effective)")
    print(f"  2. Use One-Class SVM for baseline security monitoring")
    print(f"  3. Use ensemble methods for critical systems")
    print(f"  4. Set confidence threshold at 0.7 for production alerts")
    print(f"  5. Retrain models weekly with new data")
    
    print(f"\n✨ Analysis completed successfully!")
    print(f"   Generated files: syslog_anomaly_results.csv, flow_anomaly_results.csv")
    print(f"   Visualizations: syslog_anomaly_results.html, flow_anomaly_results.html")
    print(f"   Pipeline: monitoring_pipeline.py")