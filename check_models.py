#!/usr/bin/env python
"""
Diagnostic script to check ML models, their feature counts,
and ensure consistency between training and prediction features.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules (make sure you run from project root)
from crypto_detector.core.detector import CryptoActivityDetector
from crypto_detector.core.adaptive_detector import AdaptiveCryptoDetector


def check_model_features(model_dir="historical_data/models"):
    """
    Check all saved models and analyze their feature counts and pipeline components
    """
    print(f"Checking models in directory: {model_dir}")

    # Check if directory exists
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory {model_dir} does not exist!")
        return

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

    if not model_files:
        print("No model files found!")
        return

    print(f"Found {len(model_files)} model files")

    # Load and check each model
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        print(f"\nAnalyzing model: {model_file}")

        try:
            # Load the model
            model = joblib.load(model_path)

            # Check if it's a pipeline
            if isinstance(model, Pipeline):
                print("Model is a scikit-learn Pipeline")

                # Check each step in the pipeline
                for step_name, step_model in model.named_steps.items():
                    print(f"  Step: {step_name} - {type(step_model).__name__}")

                    # Special handling for StandardScaler
                    if step_name == 'scaler' and hasattr(step_model, 'n_features_in_'):
                        print(f"  StandardScaler expects {step_model.n_features_in_} features")

                        # Check if feature names are available
                        if hasattr(step_model, 'feature_names_in_'):
                            print(f"  Feature names: {step_model.feature_names_in_.tolist()}")

                    # Check predictor model
                    if step_name == 'model' and hasattr(step_model, 'n_features_in_'):
                        print(f"  Model expects {step_model.n_features_in_} features")

                        # For tree-based models, check feature importance
                        if hasattr(step_model, 'feature_importances_'):
                            print(f"  Feature importance shape: {step_model.feature_importances_.shape}")

                            # Print top features by importance
                            if hasattr(step_model, 'feature_names_in_'):
                                feature_names = step_model.feature_names_in_
                                feature_importance = step_model.feature_importances_
                                sorted_idx = feature_importance.argsort()[::-1]
                                print("  Top 5 features by importance:")
                                for i in range(min(5, len(sorted_idx))):
                                    idx = sorted_idx[i]
                                    print(f"    {feature_names[idx]}: {feature_importance[idx]:.4f}")
            else:
                print(f"Model is a {type(model).__name__} (not a Pipeline)")

                # Check direct model features
                if hasattr(model, 'n_features_in_'):
                    print(f"Model expects {model.n_features_in_} features")

        except Exception as e:
            print(f"ERROR analyzing model {model_file}: {str(e)}")


def check_feature_extraction(model_dir="historical_data/models"):
    """
    Test feature extraction logic with a mock result to ensure consistency
    """
    print("\nTesting feature extraction logic...")

    # Create a mock detector
    detector = CryptoActivityDetector()

    # Create adaptive detector (will load models if available)
    adaptive_detector = AdaptiveCryptoDetector(detector, model_dir=model_dir)

    # Create a mock result with all necessary sections
    mock_result = {
        'symbol': 'BTC/USDT',
        'timestamp': '2025-03-19T12:00:00',
        'probability_score': 0.5,
        'signals': [
            {
                'name': 'Аномальний обсяг торгів',
                'description': 'Поточний обсяг перевищує середній на 50.00%',
                'weight': 0.35
            },
            {
                'name': 'Активна цінова динаміка',
                'description': 'Зміна ціни за останній період: 2.50%',
                'weight': 0.25
            }
        ],
        'raw_data': {
            'volume': {
                'unusual_volume': True,
                'z_score': 2.5,
                'anomaly_count': 2,
                'recent_volume': 1000,
                'mean_volume': 500,
                'percent_change': 50.0,
                'volume_acceleration': 0.15
            },
            'price': {
                'price_action_signal': True,
                'price_change_1h': 2.5,
                'price_change_24h': 5.0,
                'volatility_ratio': 1.5,
                'large_candles': 2,
                'consecutive_up': 3,
                'price_acceleration': 0.02
            },
            'order_book': {
                'order_book_signal': True,
                'buy_sell_ratio': 1.3,
                'buy_volume': 1200,
                'sell_volume': 900,
                'top_concentration': 0.7,
                'spread': 0.2,
                'has_buy_wall': True,
                'has_sell_wall': False
            },
            'social': {
                'social_signal': True,
                'mentions': 500,
                'average_mentions': 300,
                'percent_change': 66.7,
                'growth_acceleration': 0.4
            },
            'time_pattern': {
                'time_pattern_signal': True,
                'is_high_risk_hour': True,
                'is_weekend': False,
                'time_risk_score': 0.7
            },
            'correlation': {
                'correlation_signal': True,
                'correlated_coins': ['ETH/USDT', 'BNB/USDT'],
                'correlation_type': 'pump_group'
            }
        }
    }

    # Extract features
    features = adaptive_detector._extract_features_from_result(mock_result)

    print(f"Extracted {len(features)} features from mock result")
    print("Feature values:")

    # Define the expected feature names for reference
    feature_names = [
        'volume_percent_change', 'volume_z_score', 'volume_anomaly_count', 'volume_acceleration',
        'price_change_1h', 'price_change_24h', 'volatility_ratio', 'large_candles', 'consecutive_up',
        'price_acceleration',
        'buy_sell_ratio', 'top_concentration', 'has_buy_wall', 'has_sell_wall',
        'social_percent_change', 'social_growth_acceleration',
        'time_risk_score', 'is_high_risk_hour', 'is_weekend',
        'correlation_signal'
    ]

    # Print features with their names if possible
    for i, val in enumerate(features):
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        print(f"  {i}: {name} = {val}")

    # Now test with actual models
    for category in adaptive_detector.ml_models:
        print(f"\nTesting prediction with category '{category}' model")
        try:
            # Extract features and prepare for model
            features_array = np.array(features).reshape(1, -1)

            # Get the model
            model = adaptive_detector.ml_models[category]

            # Check model's expected feature count
            if hasattr(model, 'n_features_in_'):
                print(f"Model expects {model.n_features_in_} features")
            elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model', None), 'n_features_in_'):
                print(f"Model expects {model.named_steps['model'].n_features_in_} features")

            # Check if features array matches expected size
            expected_features = 16  # This is our target
            if features_array.shape[1] != expected_features:
                print(f"WARNING: Feature count mismatch - got {features_array.shape[1]}, expected {expected_features}")

                # Adjust features to match
                if features_array.shape[1] > expected_features:
                    features_array = features_array[:, :expected_features]
                    print(f"Trimmed features to {features_array.shape}")
                else:
                    features_array = np.pad(features_array, ((0, 0), (0, expected_features - features_array.shape[1])),
                                            'constant')
                    print(f"Padded features to {features_array.shape}")

            # Try prediction
            probability = model.predict_proba(features_array)[0][1]
            print(f"Prediction successful: probability = {probability:.4f}")

        except Exception as e:
            print(f"ERROR during prediction: {str(e)}")
            import traceback
            traceback.print_exc()


def rebuild_models_with_standard_features(model_dir="historical_data/models", data_dir="historical_data/ml_data"):
    """
    Rebuild models ensuring they all use the standard set of 16 features
    """
    print("\nRebuilding models with standardized features...")

    # Check directories
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory {model_dir} does not exist!")
        return

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory {data_dir} does not exist!")
        return

    # Find training data files
    data_files = [f for f in os.listdir(data_dir) if f.endswith('_ml_data.csv') or f.endswith('_training_data.csv')]

    if not data_files:
        print("No training data files found!")
        return

    print(f"Found {len(data_files)} training data files")

    # Standard feature set (16 features)
    standard_features = [
        'volume_percent_change', 'volume_z_score', 'volume_anomaly_count', 'volume_acceleration',
        'price_change_1h', 'price_change_24h', 'volatility_ratio', 'large_candles', 'consecutive_up',
        'price_acceleration',
        'buy_sell_ratio', 'top_concentration', 'has_buy_wall', 'has_sell_wall',
        'social_percent_change', 'social_growth_acceleration',
    ]

    # Process each file
    for data_file in data_files:
        category = data_file.split('_')[0].replace('adaptive', '')
        if not category:
            category = 'other'

        print(f"\nRebuilding model for category '{category}' using {data_file}")

        try:
            # Load the data
            data_path = os.path.join(data_dir, data_file)
            df = pd.read_csv(data_path)

            # Check if it has the necessary columns
            if 'is_event' not in df.columns:
                print(f"ERROR: No 'is_event' column in {data_file}")
                continue

            # Create standardized features dataframe
            features_df = pd.DataFrame(index=df.index)

            # Fill with available features
            for feature in standard_features:
                if feature in df.columns:
                    features_df[feature] = df[feature]
                else:
                    # Try alternative column names
                    alt_names = [col for col in df.columns if feature.lower() in col.lower()]
                    if alt_names:
                        features_df[feature] = df[alt_names[0]]
                    else:
                        # Fill with zeros if feature not found
                        features_df[feature] = 0
                        print(f"  WARNING: Feature '{feature}' not found, using zeros")

            # Get target variable
            y = df['is_event'].values
            X = features_df.values

            print(f"  Prepared dataset with {X.shape[1]} features and {len(y)} samples")

            # Check if we have enough samples
            if len(y) < 10 or np.sum(y) < 2 or np.sum(y == 0) < 2:
                print(f"  Not enough samples (total={len(y)}, positive={np.sum(y)}, negative={np.sum(y == 0)})")
                continue

            # Create and train pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                ))
            ])

            # Fit the pipeline
            pipeline.fit(X, y)

            # Save the model
            model_path = os.path.join(model_dir, f"{category}_model.pkl")
            joblib.dump(pipeline, model_path)

            print(f"  Model saved to {model_path}")

            # Verify model
            loaded_model = joblib.load(model_path)
            if hasattr(loaded_model, 'named_steps') and 'model' in loaded_model.named_steps:
                print(f"  Verified: model expects {loaded_model.named_steps['model'].n_features_in_} features")

        except Exception as e:
            print(f"ERROR rebuilding model for {category}: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main function to run all diagnostics
    """
    # Default paths
    model_dir = "historical_data/models"
    data_dir = "historical_data/ml_data"

    # Allow command-line overrides
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    if len(sys.argv) > 2:
        data_dir = sys.argv[2]

    print("=" * 60)
    print("CRYPTO DETECTOR MODEL DIAGNOSTICS")
    print("=" * 60)

    # Check existing models
    check_model_features(model_dir)

    # Test feature extraction
    check_feature_extraction(model_dir)

    # Ask for confirmation before rebuilding models
    response = input("\nDo you want to attempt rebuilding models with standardized features? (y/n): ")

    if response.lower() == 'y':
        rebuild_models_with_standard_features(model_dir, data_dir)

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()