#!/usr/bin/env python3
"""
Syslog Data Classification System - Fixed Version
Advanced machine learning classification for network syslog events
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

class SyslogClassifier:
    """
    Comprehensive syslog event classification system with robust error handling
    """
    
    def __init__(self, data_path='cleaned_output/all_cleaned.json'):
        """
        Initialize the classifier with cleaned syslog data
        
        Args:
            data_path: Path to the cleaned JSON syslog data
        """
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.feature_names = []
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        
    def load_data(self):
        """Load and prepare the cleaned syslog data"""
        print("Loading syslog data...")
        
        try:
            with open(self.data_path, 'r') as f:
                raw_data = json.load(f)
            
            self.data = pd.DataFrame(raw_data)
            print(f"Loaded {len(self.data)} syslog messages")
            
            # Display basic data info
            print("\nData Overview:")
            print(f"Columns: {list(self.data.columns)}")
            print(f"Data shape: {self.data.shape}")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Could not find data file at {self.data_path}")
            print("Please ensure the syslog cleaner has been run first.")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\nDataset Statistics:")
        print(f"Total messages: {len(self.data)}")
        print(f"Unique hostnames: {self.data['hostname'].nunique() if 'hostname' in self.data.columns else 'N/A'}")
        if 'timestamp' in self.data.columns:
            print(f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        
        # Event category distribution
        if 'event_category' in self.data.columns:
            print("\nEvent Category Distribution:")
            category_counts = self.data['event_category'].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(self.data)) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Device type distribution
        if 'device_type' in self.data.columns:
            print("\nDevice Type Distribution:")
            device_counts = self.data['device_type'].value_counts()
            for device, count in device_counts.items():
                percentage = (count / len(self.data)) * 100
                print(f"  {device}: {count} ({percentage:.1f}%)")
        
        # Severity distribution
        if 'severity_name' in self.data.columns:
            print("\nSeverity Distribution:")
            severity_counts = self.data['severity_name'].value_counts()
            for severity, count in severity_counts.items():
                percentage = (count / len(self.data)) * 100
                print(f"  {severity}: {count} ({percentage:.1f}%)")
        
        # Risk score statistics
        if 'risk_score' in self.data.columns:
            print(f"\nRisk Score Statistics:")
            print(f"  Mean: {self.data['risk_score'].mean():.1f}")
            print(f"  Median: {self.data['risk_score'].median():.1f}")
            print(f"  Max: {self.data['risk_score'].max()}")
            print(f"  Min: {self.data['risk_score'].min()}")
        
        # Create visualizations
        self._create_visualizations()
    
    def _create_visualizations(self):
        """Create visualizations for data exploration"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Syslog Data Analysis Dashboard', fontsize=16)
            
            # Event category distribution
            if 'event_category' in self.data.columns:
                category_counts = self.data['event_category'].value_counts()
                axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
                axes[0, 0].set_title('Event Categories')
            else:
                axes[0, 0].text(0.5, 0.5, 'Event Category\nData Not Available', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Event Categories')
            
            # Device type distribution
            if 'device_type' in self.data.columns:
                device_counts = self.data['device_type'].value_counts()
                axes[0, 1].bar(device_counts.index, device_counts.values)
                axes[0, 1].set_title('Device Types')
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'Device Type\nData Not Available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Device Types')
            
            # Severity distribution
            if 'severity_name' in self.data.columns:
                severity_counts = self.data['severity_name'].value_counts()
                axes[0, 2].bar(severity_counts.index, severity_counts.values, color='orange')
                axes[0, 2].set_title('Severity Levels')
                axes[0, 2].tick_params(axis='x', rotation=45)
            else:
                axes[0, 2].text(0.5, 0.5, 'Severity\nData Not Available', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Severity Levels')
            
            # Risk score distribution
            if 'risk_score' in self.data.columns:
                axes[1, 0].hist(self.data['risk_score'], bins=20, color='red', alpha=0.7)
                axes[1, 0].set_title('Risk Score Distribution')
                axes[1, 0].set_xlabel('Risk Score')
                axes[1, 0].set_ylabel('Frequency')
            else:
                axes[1, 0].text(0.5, 0.5, 'Risk Score\nData Not Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Risk Score Distribution')
            
            # Hourly event distribution
            if 'timestamp' in self.data.columns:
                try:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data['hour'] = self.data['timestamp'].dt.hour
                    hourly_counts = self.data['hour'].value_counts().sort_index()
                    axes[1, 1].plot(hourly_counts.index, hourly_counts.values, marker='o')
                    axes[1, 1].set_title('Hourly Event Distribution')
                    axes[1, 1].set_xlabel('Hour of Day')
                    axes[1, 1].set_ylabel('Event Count')
                except:
                    axes[1, 1].text(0.5, 0.5, 'Timestamp\nParsing Failed', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Hourly Event Distribution')
            else:
                axes[1, 1].text(0.5, 0.5, 'Timestamp\nData Not Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Hourly Event Distribution')
            
            # Security vs non-security events
            if 'is_security' in self.data.columns:
                security_counts = self.data['is_security'].value_counts()
                labels = ['Non-Security', 'Security'] if False in security_counts.index else ['Security']
                axes[1, 2].pie(security_counts.values, labels=labels, 
                              colors=['lightblue', 'red'], autopct='%1.1f%%')
                axes[1, 2].set_title('Security vs Non-Security Events')
            else:
                axes[1, 2].text(0.5, 0.5, 'Security Flag\nData Not Available', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Security vs Non-Security Events')
            
            plt.tight_layout()
            plt.savefig('syslog_analysis_dashboard.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("\nVisualization saved as 'syslog_analysis_dashboard.png'")
            
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    def prepare_features(self, target_column='event_category'):
        """
        Prepare features for machine learning classification with robust error handling
        
        Args:
            target_column: Column to use as the target variable
        """
        print(f"\nPreparing features for classification (target: {target_column})...")
        
        # First, let's check what columns are actually available
        print(f"Available columns: {list(self.data.columns)}")
        
        # Define potential features and check which ones exist
        potential_numerical_features = [
            'facility', 'severity', 'priority', 'pid', 'risk_score',
            'destination_port', 'source_port', 'as_number'
        ]
        
        potential_boolean_features = [
            'is_security', 'is_authentication', 'is_network_event', 'is_system_event'
        ]
        
        potential_categorical_features = [
            'device_type', 'program', 'severity_name', 'protocol'
        ]
        
        # Filter to only include columns that actually exist
        available_numerical = [col for col in potential_numerical_features if col in self.data.columns]
        available_boolean = [col for col in potential_boolean_features if col in self.data.columns]
        available_categorical = [col for col in potential_categorical_features if col in self.data.columns]
        
        print(f"Available numerical features: {available_numerical}")
        print(f"Available boolean features: {available_boolean}")
        print(f"Available categorical features: {available_categorical}")
        
        # Create feature matrix
        features_list = []
        
        # Add numerical and boolean features
        all_numeric_features = available_numerical + available_boolean
        if all_numeric_features:
            try:
                numerical_data = self.data[all_numeric_features].fillna(0)
                # Convert boolean columns to int if they're boolean
                for col in available_boolean:
                    if col in numerical_data.columns:
                        numerical_data[col] = numerical_data[col].astype(int)
                features_list.append(numerical_data)
                print(f"Added {len(all_numeric_features)} numerical/boolean features")
            except Exception as e:
                print(f"Warning: Could not process numerical features: {e}")
                # Create dummy numerical features
                dummy_numerical = pd.DataFrame({'dummy_numerical': [0] * len(self.data)})
                features_list.append(dummy_numerical)
        else:
            print("Warning: No numerical features found, creating dummy features")
            dummy_numerical = pd.DataFrame({'dummy_numerical': [0] * len(self.data)})
            features_list.append(dummy_numerical)
        
        # One-hot encode categorical features
        for col in available_categorical:
            try:
                # Fill missing values with 'unknown'
                col_data = self.data[col].fillna('unknown')
                dummies = pd.get_dummies(col_data, prefix=col)
                features_list.append(dummies)
                print(f"Added categorical feature: {col} ({dummies.shape[1]} dummy variables)")
            except Exception as e:
                print(f"Warning: Could not encode {col}: {e}")
        
        # TF-IDF features from message text
        if 'message' in self.data.columns:
            try:
                message_text = self.data['message'].fillna('').astype(str)
                message_tfidf = self.tfidf_vectorizer.fit_transform(message_text)
                message_df = pd.DataFrame(
                    message_tfidf.toarray(), 
                    columns=[f'tfidf_{i}' for i in range(message_tfidf.shape[1])]
                )
                features_list.append(message_df)
                print(f"Added TF-IDF features: {message_df.shape[1]} text features")
            except Exception as e:
                print(f"Warning: Could not create TF-IDF features: {e}")
                # Create dummy TF-IDF features
                dummy_tfidf = pd.DataFrame({f'tfidf_{i}': [0] * len(self.data) for i in range(10)})
                features_list.append(dummy_tfidf)
                print("Added dummy TF-IDF features")
        else:
            print("Warning: No message column found, creating dummy text features")
            dummy_tfidf = pd.DataFrame({f'tfidf_{i}': [0] * len(self.data) for i in range(10)})
            features_list.append(dummy_tfidf)
        
        # Time-based features
        if 'timestamp' in self.data.columns:
            try:
                timestamp_series = pd.to_datetime(self.data['timestamp'], errors='coerce')
                
                time_features = pd.DataFrame({
                    'hour': timestamp_series.dt.hour.fillna(12),
                    'day_of_week': timestamp_series.dt.dayofweek.fillna(0),
                    'is_weekend': (timestamp_series.dt.dayofweek >= 5).astype(int).fillna(0),
                    'is_business_hours': ((timestamp_series.dt.hour >= 8) & 
                                         (timestamp_series.dt.hour <= 18)).astype(int).fillna(1)
                })
                features_list.append(time_features)
                print(f"Added time-based features: {time_features.shape[1]} temporal features")
            except Exception as e:
                print(f"Warning: Could not create time features: {e}")
                # Create dummy time features
                dummy_time = pd.DataFrame({
                    'hour': [12] * len(self.data),
                    'day_of_week': [0] * len(self.data),
                    'is_weekend': [0] * len(self.data),
                    'is_business_hours': [1] * len(self.data)
                })
                features_list.append(dummy_time)
                print("Added dummy time features")
        else:
            print("Warning: No timestamp column found, creating dummy time features")
            dummy_time = pd.DataFrame({
                'hour': [12] * len(self.data),
                'day_of_week': [0] * len(self.data),
                'is_weekend': [0] * len(self.data),
                'is_business_hours': [1] * len(self.data)
            })
            features_list.append(dummy_time)
        
        # Combine all features
        if features_list:
            try:
                self.X = pd.concat(features_list, axis=1)
                self.feature_names = list(self.X.columns)
                print(f"Successfully combined features into matrix of shape: {self.X.shape}")
            except Exception as e:
                print(f"Error combining features: {e}")
                return False
        else:
            print("Error: No features could be created!")
            return False
        
        # Prepare target variable
        if target_column not in self.data.columns:
            print(f"Error: Target column '{target_column}' not found in data!")
            print(f"Available columns: {list(self.data.columns)}")
            return False
        
        self.y = self.data[target_column]
        
        # Remove any rows with missing target values
        valid_mask = ~self.y.isna()
        self.X = self.X[valid_mask]
        self.y = self.y[valid_mask]
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Target classes: {self.y.unique()}")
        print(f"Target distribution:")
        target_counts = self.y.value_counts()
        for target, count in target_counts.items():
            percentage = (count / len(self.y)) * 100
            print(f"  {target}: {count} ({percentage:.1f}%)")
        
        # Split data
        try:
            # Try stratified split first
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
        except Exception as e:
            print(f"Warning: Stratified split failed ({e}), using regular split")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        return True
    
    def train_classifiers(self):
        """Train multiple classification algorithms"""
        print("\n" + "="*60)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*60)
        
        if self.X is None or self.y is None:
            print("Error: Features not prepared. Please run prepare_features() first.")
            return {}
        
        # Define classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000
            )
        }
        
        # Only add SVM and Naive Bayes if dataset is not too large
        if len(self.X_train) < 10000:
            classifiers['Support Vector Machine'] = SVC(random_state=42, probability=True)
            classifiers['Naive Bayes'] = MultinomialNB()
        
        # Train and evaluate each classifier
        results = {}
        
        for name, classifier in classifiers.items():
            print(f"\nTraining {name}...")
            
            try:
                # Handle data preprocessing for specific algorithms
                X_train_processed = self.X_train.copy()
                X_test_processed = self.X_test.copy()
                
                # For algorithms that need non-negative features (Naive Bayes)
                if name == 'Naive Bayes':
                    # Check if features are already non-negative
                    if (X_train_processed < 0).any().any():
                        # Make features non-negative for Naive Bayes
                        scaler = StandardScaler()
                        X_train_processed = scaler.fit_transform(X_train_processed)
                        X_test_processed = scaler.transform(X_test_processed)
                        # Shift to make non-negative
                        X_train_processed = X_train_processed - X_train_processed.min() + 1
                        X_test_processed = X_test_processed - X_test_processed.min() + 1
                    else:
                        # Features are already non-negative, just ensure no zeros for MultinomialNB
                        X_train_processed = X_train_processed + 1e-10
                        X_test_processed = X_test_processed + 1e-10
                
                # Train the model
                classifier.fit(X_train_processed, self.y_train)
                
                # Make predictions
                y_pred = classifier.predict(X_test_processed)
                y_pred_proba = None
                
                if hasattr(classifier, 'predict_proba'):
                    y_pred_proba = classifier.predict_proba(X_test_processed)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # Cross-validation score
                try:
                    cv_scores = cross_val_score(classifier, X_train_processed, self.y_train, cv=3)
                except:
                    cv_scores = np.array([accuracy])  # Fallback if CV fails
                
                results[name] = {
                    'model': classifier,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
        
        self.models = results
        return results
    
    def evaluate_models(self):
        """Detailed evaluation of trained models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        if not self.models:
            print("No models trained yet. Please run train_classifiers() first.")
            return None, None
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in self.models.items():
            comparison_data.append({
                'Model': name,
                'Test Accuracy': result['accuracy'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Detailed evaluation for best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model_result = self.models[best_model_name]
        
        print(f"\nDetailed Evaluation for Best Model: {best_model_name}")
        print("="*50)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, best_model_result['predictions']))
        
        # Confusion matrix
        self._plot_confusion_matrix(
            self.y_test, 
            best_model_result['predictions'], 
            f"{best_model_name} Confusion Matrix"
        )
        
        # Feature importance (if available)
        if hasattr(best_model_result['model'], 'feature_importances_'):
            self._plot_feature_importance(
                best_model_result['model'], 
                best_model_name
            )
        
        # Model comparison visualization
        self._plot_model_comparison(comparison_df)
        
        return best_model_name, best_model_result
    
    def _plot_confusion_matrix(self, y_true, y_pred, title):
        """Plot confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=np.unique(y_true),
                       yticklabels=np.unique(y_true))
            plt.title(title)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create confusion matrix plot: {e}")
    
    def _plot_feature_importance(self, model, model_name):
        """Plot feature importance for tree-based models"""
        try:
            if not hasattr(model, 'feature_importances_'):
                return
            
            # Get top 20 features
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'{model_name.lower().replace(" ", "_")}_feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nTop 10 Most Important Features ({model_name}):")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        except Exception as e:
            print(f"Could not create feature importance plot: {e}")
    
    def _plot_model_comparison(self, comparison_df):
        """Plot model performance comparison"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy comparison
            sns.barplot(data=comparison_df, x='Test Accuracy', y='Model', ax=axes[0])
            axes[0].set_title('Test Accuracy Comparison')
            axes[0].set_xlabel('Accuracy')
            
            # Cross-validation score comparison
            axes[1].errorbar(comparison_df['CV Mean'], comparison_df['Model'], 
                            xerr=comparison_df['CV Std'], fmt='o', capsize=5)
            axes[1].set_title('Cross-Validation Score Comparison')
            axes[1].set_xlabel('CV Score (mean ± std)')
            
            plt.tight_layout()
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create model comparison plot: {e}")
    
    def save_results(self, output_file='classification_results.json'):
        """Save classification results to file"""
        try:
            results = {
                'model_performance': {},
                'feature_importance': {},
                'data_insights': {
                    'total_events': len(self.data),
                    'unique_categories': list(self.y.unique()) if self.y is not None else [],
                    'category_distribution': self.y.value_counts().to_dict() if self.y is not None else {}
                }
            }
            
            # Save model performance
            for name, result in self.models.items():
                results['model_performance'][name] = {
                    'accuracy': float(result['accuracy']),
                    'cv_mean': float(result['cv_mean']),
                    'cv_std': float(result['cv_std'])
                }
            
            # Save feature importance for tree-based models
            for name, result in self.models.items():
                if hasattr(result['model'], 'feature_importances_'):
                    feature_importance = dict(zip(
                        self.feature_names, 
                        result['model'].feature_importances_
                    ))
                    results['feature_importance'][name] = {
                        k: float(v) for k, v in feature_importance.items()
                    }
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to {output_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Main execution function"""
    print("="*70)
    print("SYSLOG EVENT CLASSIFICATION SYSTEM")
    print("="*70)
    
    # Initialize classifier
    classifier = SyslogClassifier()
    
    # Load and explore data
    if not classifier.load_data():
        return
    
    classifier.explore_data()
    
    # Prepare features for classification
    if not classifier.prepare_features(target_column='event_category'):
        print("Failed to prepare features. Exiting.")
        return
    
    # Train multiple classification models
    results = classifier.train_classifiers()
    
    if not results:
        print("No models were successfully trained. Exiting.")
        return
    
    # Evaluate models and get the best one
    best_model_name, best_result = classifier.evaluate_models()
    
    if best_model_name:
        # Save results
        classifier.save_results()
        
        print("\n" + "="*70)
        print("CLASSIFICATION ANALYSIS COMPLETE!")
        print("="*70)
        print(f"Best performing model: {best_model_name}")
        print(f"Best accuracy: {best_result['accuracy']:.4f}")
        print("\nGenerated files:")
        print("  - syslog_analysis_dashboard.png")
        print("  - model_comparison.png")
        print("  - confusion matrix plots")
        print("  - feature importance plots (if available)")
        print("  - classification_results.json")
        print("\nRecommendations:")
        print("  1. Use the best performing model for production deployment")
        print("  2. Monitor feature importance for model interpretation")
        print("  3. Retrain models periodically with new data")
        print("  4. Consider ensemble methods for improved performance")
    else:
        print("No successful model evaluation completed.")


if __name__ == "__main__":
    main()