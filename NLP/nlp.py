#!/usr/bin/env python3
"""
NLP for Network Operations - Demonstration Code
Processes syslog entries for automated classification and analysis
Optimized for CPU-only execution with 20-22k log entries
"""

import pandas as pd
import numpy as np
import re
import pickle
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Core NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Sentiment analysis
from textblob import TextBlob

# Download required NLTK data (run once)
def download_nltk_data():
    """Download required NLTK data with error handling"""
    try:
        # Try punkt_tab first (newer NLTK versions)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading punkt_tab...")
            nltk.download('punkt_tab')
    except:
        # Fallback to punkt (older NLTK versions)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading punkt...")
            nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords')

# Download NLTK data
download_nltk_data()

class NetworkLogNLP:
    """
    Network Log Natural Language Processing System
    Demonstrates log classification, sentiment analysis, and pattern recognition
    """
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        
        # Initialize stop words with error handling
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stop words if NLTK stopwords not available
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                'with', 'through', 'during', 'before', 'after', 'above', 'below', 
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once'
            }
        
        # Network-specific stop words to remove
        self.network_stop_words = {
            'interface', 'config', 'system', 'device', 'network', 
            'connection', 'status', 'message', 'log', 'entry'
        }
        
        # Classification models
        self.severity_classifier = None
        self.category_classifier = None
        self.vectorizer = None
        
        # Pattern recognition
        self.ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        self.interface_pattern = re.compile(r'(?:Gi|Fa|Te|Se|Et|Lo)\d+(?:/\d+)*', re.IGNORECASE)
        self.error_code_pattern = re.compile(r'%\w+-\d+-\w+')
        
    def generate_sample_data(self, n_samples=22000):
        """
        Generate realistic network syslog data for demonstration
        Simulates the variety found in enterprise networks
        """
        print(f"Generating {n_samples} sample syslog entries...")
        
        # Define realistic log message templates
        templates = {
            'interface_down': [
                "Interface {interface} changed state to down",
                "Line protocol on Interface {interface}, changed state to down",
                "{interface}: link is down",
                "Interface {interface} is down, line protocol is down"
            ],
            'interface_up': [
                "Interface {interface} changed state to up",
                "Line protocol on Interface {interface}, changed state to up", 
                "{interface}: link is up",
                "Interface {interface} is up, line protocol is up"
            ],
            'bgp_issues': [
                "BGP: {ip} went from Established to Idle",
                "BGP: {ip} connection closed",
                "BGP neighbor {ip} Down",
                "BGP: peer {ip} flapping detected"
            ],
            'authentication': [
                "Authentication failed for user {user} from {ip}",
                "Login failed for {user} from {ip}",
                "Invalid password for user {user}",
                "User {user} authentication successful from {ip}"
            ],
            'security': [
                "Denied connection from {ip} to port 22",
                "Firewall blocked connection from {ip}",
                "Intrusion attempt detected from {ip}",
                "Security violation: unauthorized access from {ip}"
            ],
            'performance': [
                "High CPU utilization detected: {percent}%",
                "Memory usage critical: {percent}%",
                "Interface {interface} bandwidth utilization high",
                "Queue overflow on interface {interface}"
            ],
            'configuration': [
                "Configuration changed by user {user}",
                "VLAN {vlan} configuration updated",
                "Routing table updated",
                "ACL configuration modified"
            ]
        }
        
        # Sample data for placeholders
        interfaces = ['Gi0/1', 'Gi0/2', 'Fa0/1', 'Te1/0/1', 'Se0/0/0', 'Et0/1']
        ips = ['192.168.1.100', '10.0.0.50', '172.16.1.200', '203.0.113.5', '198.51.100.10']
        users = ['admin', 'jsmith', 'network_ops', 'backup_user', 'monitoring']
        vlans = ['100', '200', '300', '999']
        
        logs = []
        
        for i in range(n_samples):
            # Random timestamp within last 30 days
            timestamp = datetime.now() - timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            # Random severity level
            severity_weights = [0.4, 0.3, 0.15, 0.1, 0.05]  # Most logs are info/warning
            severity = np.random.choice(['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG'], 
                                      p=severity_weights)
            
            # Random device
            device = f"router-{np.random.randint(1, 20):02d}"
            
            # Choose random category and template
            category = np.random.choice(list(templates.keys()))
            template = np.random.choice(templates[category])
            
            # Fill template with random data
            message = template.format(
                interface=np.random.choice(interfaces),
                ip=np.random.choice(ips),
                user=np.random.choice(users),
                vlan=np.random.choice(vlans),
                percent=np.random.randint(70, 99)
            )
            
            # Add error codes for some entries
            if np.random.random() < 0.3:
                error_code = f"%{device.upper()}-{np.random.randint(1,6)}-{np.random.choice(['CONFIG', 'LINK', 'AUTH', 'SYS'])}"
                message = f"{error_code}: {message}"
            
            logs.append({
                'timestamp': timestamp,
                'device': device,
                'severity': severity,
                'message': message,
                'category': category
            })
        
        df = pd.DataFrame(logs)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Generated {len(df)} log entries")
        print(f"Categories: {df['category'].value_counts().to_dict()}")
        print(f"Severity levels: {df['severity'].value_counts().to_dict()}")
        
        return df
    
    def preprocess_text(self, text):
        """
        Advanced text preprocessing for network logs
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Extract and preserve important network entities
        ips = self.ip_pattern.findall(text)
        interfaces = self.interface_pattern.findall(text)
        error_codes = self.error_code_pattern.findall(text)
        
        # Replace entities with standardized tokens
        text = self.ip_pattern.sub('IP_ADDRESS', text)
        text = self.interface_pattern.sub('INTERFACE_NAME', text)
        text = self.error_code_pattern.sub('ERROR_CODE', text)
        
        # Remove special characters but keep important network symbols
        text = re.sub(r'[^\w\s\-_/:]', ' ', text)
        
        # Tokenize with error handling
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split if NLTK tokenizer fails
            tokens = text.split()
        
        # Remove stop words and short tokens
        tokens = [token for token in tokens if 
                 token not in self.stop_words and 
                 token not in self.network_stop_words and 
                 len(token) > 2]
        
        # Stem tokens with error handling
        try:
            tokens = [self.stemmer.stem(token) for token in tokens]
        except:
            # If stemming fails, just use original tokens
            pass
        
        return ' '.join(tokens)
    
    def extract_features(self, df):
        """
        Extract additional features from log entries
        """
        print("Extracting features from log data...")
        
        # Text features
        df['processed_message'] = df['message'].apply(self.preprocess_text)
        df['message_length'] = df['message'].str.len()
        df['word_count'] = df['processed_message'].str.split().str.len()
        
        # Network entity counts
        df['ip_count'] = df['message'].apply(lambda x: len(self.ip_pattern.findall(str(x))))
        df['interface_count'] = df['message'].apply(lambda x: len(self.interface_pattern.findall(str(x))))
        df['has_error_code'] = df['message'].apply(lambda x: bool(self.error_code_pattern.search(str(x))))
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 17) & (df['day_of_week'] < 5))
        
        # Severity encoding
        severity_map = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
        df['severity_numeric'] = df['severity'].map(severity_map)
        
        return df
    
    def train_classification_models(self, df):
        """
        Train multiple classification models
        """
        print("Training classification models...")
        
        # Prepare features
        X_text = df['processed_message']
        
        # Create TF-IDF vectorizer optimized for network logs
        self.vectorizer = TfidfVectorizer(
            max_features=5000,           # Limit features for CPU efficiency
            ngram_range=(1, 2),          # Include bigrams for context
            min_df=2,                    # Ignore rare terms
            max_df=0.95,                 # Ignore very common terms
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )
        
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        # Additional numerical features
        numerical_features = ['message_length', 'word_count', 'ip_count', 
                            'interface_count', 'hour', 'severity_numeric']
        X_numerical = df[numerical_features].values
        
        # Combine text and numerical features
        from scipy.sparse import hstack
        X_combined = hstack([X_vectorized, X_numerical])
        
        # Train category classifier
        print("Training category classifier...")
        y_category = df['category']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_category, test_size=0.2, random_state=42, stratify=y_category
        )
        
        # Use Random Forest for good CPU performance and interpretability
        self.category_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        self.category_classifier.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.category_classifier.predict(X_test)
        print("\nCategory Classification Results:")
        print(classification_report(y_test, y_pred))
        
        # Train severity classifier
        print("\nTraining severity classifier...")
        y_severity = df['severity']
        
        X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
            X_combined, y_severity, test_size=0.2, random_state=42, stratify=y_severity
        )
        
        self.severity_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        self.severity_classifier.fit(X_train_sev, y_train_sev)
        
        # Evaluate severity model
        y_pred_sev = self.severity_classifier.predict(X_test_sev)
        print("\nSeverity Classification Results:")
        print(classification_report(y_test_sev, y_pred_sev))
        
        return X_test, y_test, y_pred
    
    def analyze_sentiment(self, df):
        """
        Perform sentiment analysis on log messages
        Useful for user-facing logs and error messages
        """
        print("Analyzing sentiment in log messages...")
        
        def get_sentiment(text):
            try:
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    return 'positive'
                elif polarity < -0.1:
                    return 'negative'
                else:
                    return 'neutral'
            except:
                return 'neutral'
        
        df['sentiment'] = df['message'].apply(get_sentiment)
        df['sentiment_score'] = df['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        
        sentiment_counts = df['sentiment'].value_counts()
        print(f"Sentiment distribution: {sentiment_counts.to_dict()}")
        
        # Identify most negative messages (potential user impact)
        negative_logs = df[df['sentiment'] == 'negative'].nlargest(5, 'sentiment_score', keep='all')
        print("\nMost concerning log messages (negative sentiment):")
        for idx, row in negative_logs.iterrows():
            print(f"- {row['message'][:100]}...")
        
        return df
    
    def detect_patterns(self, df):
        """
        Advanced pattern detection in network logs
        """
        print("Detecting patterns and anomalies...")
        
        # Frequency analysis
        device_activity = df['device'].value_counts()
        print(f"\nTop 5 most active devices:")
        print(device_activity.head())
        
        # Time-based patterns
        hourly_activity = df.groupby('hour').size()
        peak_hours = hourly_activity.nlargest(3)
        print(f"\nPeak activity hours: {peak_hours.to_dict()}")
        
        # Error clustering
        error_messages = df[df['severity'].isin(['ERROR', 'CRITICAL'])]
        if not error_messages.empty:
            error_patterns = error_messages.groupby('category').size()
            print(f"\nError distribution by category: {error_patterns.to_dict()}")
        
        # Interface flapping detection
        interface_changes = df[df['message'].str.contains('changed state', case=False, na=False)]
        if not interface_changes.empty:
            flapping_interfaces = interface_changes.groupby(['device', 'message']).size()
            potential_flapping = flapping_interfaces[flapping_interfaces > 2]
            if not potential_flapping.empty:
                print(f"\nPotential interface flapping detected:")
                print(potential_flapping.head())
        
        # BGP instability
        bgp_issues = df[df['message'].str.contains('bgp|BGP', case=False, na=False)]
        if not bgp_issues.empty:
            bgp_frequency = bgp_issues.groupby(bgp_issues['timestamp'].dt.hour).size()
            print(f"\nBGP issues by hour: {bgp_frequency.to_dict()}")
        
        return {
            'device_activity': device_activity,
            'hourly_activity': hourly_activity,
            'error_patterns': error_patterns if not error_messages.empty else {},
            'bgp_frequency': bgp_frequency if not bgp_issues.empty else {}
        }
    
    def predict_new_logs(self, new_messages):
        """
        Classify new log messages using trained models
        """
        if self.category_classifier is None or self.vectorizer is None:
            raise ValueError("Models not trained yet. Call train_classification_models first.")
        
        # Preprocess new messages
        processed_messages = [self.preprocess_text(msg) for msg in new_messages]
        
        # Create dummy numerical features for prediction
        dummy_features = np.zeros((len(new_messages), 6))  # 6 numerical features
        
        # Vectorize text
        X_text = self.vectorizer.transform(processed_messages)
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_text, dummy_features])
        
        # Predict categories and probabilities
        categories = self.category_classifier.predict(X_combined)
        category_probs = self.category_classifier.predict_proba(X_combined)
        
        # Predict severity
        severities = self.severity_classifier.predict(X_combined)
        severity_probs = self.severity_classifier.predict_proba(X_combined)
        
        results = []
        for i, msg in enumerate(new_messages):
            results.append({
                'message': msg,
                'predicted_category': categories[i],
                'category_confidence': max(category_probs[i]),
                'predicted_severity': severities[i], 
                'severity_confidence': max(severity_probs[i])
            })
        
        return results
    
    def generate_insights_report(self, df, patterns):
        """
        Generate comprehensive insights report
        """
        report = []
        report.append("=== NETWORK LOG ANALYSIS REPORT ===\n")
        
        # Summary statistics
        total_logs = len(df)
        date_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        report.append(f"Analysis Period: {date_range}")
        report.append(f"Total Log Entries: {total_logs:,}")
        report.append(f"Unique Devices: {df['device'].nunique()}")
        
        # Severity breakdown
        severity_dist = df['severity'].value_counts()
        report.append(f"\nSeverity Distribution:")
        for severity, count in severity_dist.items():
            percentage = (count / total_logs) * 100
            report.append(f"  {severity}: {count:,} ({percentage:.1f}%)")
        
        # Category insights
        category_dist = df['category'].value_counts()
        report.append(f"\nTop Issue Categories:")
        for category, count in category_dist.head().items():
            percentage = (count / total_logs) * 100
            report.append(f"  {category}: {count:,} ({percentage:.1f}%)")
        
        # Critical issues
        critical_logs = df[df['severity'] == 'CRITICAL']
        if not critical_logs.empty:
            report.append(f"\nCritical Issues Detected: {len(critical_logs)}")
            critical_categories = critical_logs['category'].value_counts()
            for category, count in critical_categories.items():
                report.append(f"  {category}: {count}")
        
        # Recommendations
        report.append(f"\n=== RECOMMENDATIONS ===")
        
        if severity_dist.get('ERROR', 0) + severity_dist.get('CRITICAL', 0) > total_logs * 0.1:
            report.append("• High error rate detected - investigate root causes")
        
        if 'interface_down' in category_dist and category_dist['interface_down'] > 50:
            report.append("• Frequent interface issues - check physical connections")
        
        if 'bgp_issues' in category_dist and category_dist['bgp_issues'] > 20:
            report.append("• BGP instability detected - review routing configuration")
        
        if patterns['hourly_activity'].max() > patterns['hourly_activity'].mean() * 3:
            peak_hour = patterns['hourly_activity'].idxmax()
            report.append(f"• Activity spike at hour {peak_hour} - investigate capacity")
        
        return '\n'.join(report)

def main():
    """
    Main demonstration function
    """
    print("Starting NLP Network Log Analysis Demonstration")
    print("=" * 60)
    
    # Initialize NLP system
    nlp_system = NetworkLogNLP()
    
    # Generate sample data (replace with actual data loading in production)
    print("\n1. GENERATING SAMPLE DATA")
    df = nlp_system.generate_sample_data(n_samples=22000)
    
    # Feature extraction
    print("\n2. FEATURE EXTRACTION AND PREPROCESSING")
    df = nlp_system.extract_features(df)
    
    # Train classification models
    print("\n3. TRAINING CLASSIFICATION MODELS")
    X_test, y_test, y_pred = nlp_system.train_classification_models(df)
    
    # Sentiment analysis
    print("\n4. SENTIMENT ANALYSIS")
    df = nlp_system.analyze_sentiment(df)
    
    # Pattern detection
    print("\n5. PATTERN DETECTION AND ANOMALY ANALYSIS")
    patterns = nlp_system.detect_patterns(df)
    
    # Test predictions on new messages
    print("\n6. TESTING PREDICTIONS ON NEW LOG MESSAGES")
    test_messages = [
        "Interface Gi0/1 changed state to down",
        "BGP neighbor 192.168.1.1 went from Established to Idle", 
        "Authentication failed for user admin from 10.0.0.100",
        "High CPU utilization detected: 95%",
        "Configuration changed by user network_ops"
    ]
    
    predictions = nlp_system.predict_new_logs(test_messages)
    print("\nPrediction Results:")
    for pred in predictions:
        print(f"Message: {pred['message']}")
        print(f"  Category: {pred['predicted_category']} (confidence: {pred['category_confidence']:.2f})")
        print(f"  Severity: {pred['predicted_severity']} (confidence: {pred['severity_confidence']:.2f})")
        print()
    
    # Generate comprehensive report
    print("\n7. GENERATING INSIGHTS REPORT")
    report = nlp_system.generate_insights_report(df, patterns)
    print(report)
    
    # Save model for future use
    print("\n8. SAVING TRAINED MODELS")
    model_data = {
        'category_classifier': nlp_system.category_classifier,
        'severity_classifier': nlp_system.severity_classifier,
        'vectorizer': nlp_system.vectorizer
    }
    
    with open('network_nlp_models.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Models saved to 'network_nlp_models.pkl'")
    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    main()
    