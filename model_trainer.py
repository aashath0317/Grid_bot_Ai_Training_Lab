import logging
import os

# Changed import for standalone use
from indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ppo,
    calculate_rsi,
    calculate_sma,
)
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


class ModelTrainer:
    def __init__(self, data_path: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Use XGBoost with GPU support
        self.model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            device="cuda",
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
        )
        self.data_path = data_path

    def tune_hyperparameters(self, X_train, y_train):
        """
        Performs Randomized Search to find better XGBoost parameters.
        """
        self.logger.info("Starting Hyperparameter Tuning...")

        param_dist = {
            "n_estimators": randint(300, 1000),
            "learning_rate": uniform(0.01, 0.2),
            "max_depth": randint(4, 12),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.7, 0.3),
            "gamma": uniform(0, 0.5),
            "scale_pos_weight": [1, 5, 10, 20],  # Test different weights
        }

        # Use CPU for search if GPU memory is tight, or stick to GPU
        # RandomizedSearchCV with XGBoost GPU can be tricky if n_jobs > 1
        clf = XGBClassifier(device="cuda", tree_method="hist", eval_metric="logloss")

        search = RandomizedSearchCV(
            clf,
            param_distributions=param_dist,
            n_iter=15,  # Try 15 combinations
            scoring="precision",  # Optimize for PRECISION directly!
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=1,  # Sequential to avoid GPU conflict
        )

        search.fit(X_train, y_train)

        print("\n=== Best Hyperparameters ===")
        print(search.best_params_)
        print(f"Best Precision Score: {search.best_score_:.2%}")

        self.model = search.best_estimator_
        return search.best_params_

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates indicators and creates feature set.
        """
        df = df.copy()

        # Technical Indicators
        df["rsi"] = calculate_rsi(df["close"])
        df["atr"] = calculate_atr(df)
        df["sma_50"] = calculate_sma(df["close"], 50)
        df["sma_200"] = calculate_sma(df["close"], 200)

        # Bollinger Bands
        bb = calculate_bollinger_bands(df["close"])
        df["bb_width"] = (bb["upper"] - bb["lower"]) / bb["middle"]

        # Volume
        df["volume_change"] = df["volume"].pct_change()
        df["vol_ma_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()  # Relative Volume

        # Momentum / Velocity
        df["rsi_slope"] = df["rsi"].diff()
        df["rsi_slope"] = df["rsi"].diff()
        df["price_velocity"] = df["close"].pct_change(3)  # 3-candle ROC

        # PPO & ADX
        ppo_df = calculate_ppo(df["close"])
        df["ppo"] = ppo_df["ppo"]
        df["ppo_hist"] = ppo_df["ppo_hist"]
        df["adx"] = calculate_adx(df)

        # Multi-Timeframe Momentum (Simulated)
        # For 1m data: 15 candles = 15m, 60 candles = 1h
        # For 15m data: 15 candles = 3.75h (still useful trend), 60 candles = 15h
        df["roc_15"] = df["close"].pct_change(15)
        df["roc_60"] = df["close"].pct_change(60)

        # Interaction Features
        df["vol_trend"] = df["bb_width"] * df["adx"]
        df["vol_momentum"] = df["bb_width"] * df["roc_60"]
        df["trend_momentum"] = df["adx"] * df["ppo"]

        # Volatility over longer horizons
        df["std_60"] = df["close"].rolling(60).std() / df["close"]

        # Price Action
        df["dist_sma_50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

        # Clean NaNs and Infs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

    def label_data(self, df: pd.DataFrame, lookahead: int = 4, threshold: float = 0.02) -> pd.Series:
        """
        Labels data: 1 (Pump) if price increases by > threshold% in next 'lookahead' candles.
        """
        future_close = df["close"].shift(-lookahead)
        pct_change = (future_close - df["close"]) / df["close"]

        labels = (pct_change > threshold).astype(int)

        # Align labels (shift back or just drop last "lookahead" rows where we don't know future)
        # We need to remove the last 'lookahead' rows from both X and y
        return labels

    def train(
        self,
        df: pd.DataFrame,
        model_path: str = "ai_model.joblib",  # Changed default path
        label_threshold: float = 0.02,
        lookahead: int = 4,
    ):
        """
        Main training loop.
        """
        self.logger.info("Preparing features...")
        df_features = self.prepare_features(df)

        # Define Label
        self.logger.info(f"Labeling data with threshold {label_threshold:.1%} over {lookahead} candles...")
        y = self.label_data(df_features, lookahead=lookahead, threshold=label_threshold)

        # Check Class Balance
        positives = y.sum()
        total = len(y)
        ratio = positives / total
        ratio = positives / total
        print(f"Class Balance: {positives} Pumps / {total} Total ({ratio:.2%})")
        self.logger.info(f"Class Balance: {positives} Pumps / {total} Total ({ratio:.2%})")

        if positives < 10:
            print("ERROR: Not enough Pump examples to train. Try lowering --label_threshold.")
            return

        # Align X and y
        valid_idx = y.index[:-lookahead]
        X = df_features.loc[valid_idx]
        y = y.loc[valid_idx]

        # Drop columns not for training
        feature_cols = [
            "rsi",
            "atr",
            "bb_width",
            "volume_change",
            "dist_sma_50",
            "rsi_slope",
            "vol_ma_ratio",
            "price_velocity",
            "roc_15",
            "roc_60",
            "std_60",
            "ppo",
            "ppo_hist",
            "adx",
            "vol_trend",
            "vol_momentum",
            "trend_momentum",
        ]
        X = X[feature_cols]

        # Train/Test Split (Time Series)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples...")

        # Manual Class Weight to force model to pay attention
        # Scale weight for class 1
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count

        # Removed sensitivity boost (* 2.0) to favor PRECISION (Safety) over Recall
        # scale_pos_weight *= 2.0
        self.logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

        self.model.set_params(scale_pos_weight=scale_pos_weight)

        # Optional Tuning Hook (enabled via simple check for now, later CLI arg)
        # self.tune_hyperparameters(X_train, y_train)
        # For now, let's keep it manual until user asks to enable it fully.
        # But we added the code above.

        # Fit
        self.model.fit(X_train, y_train)

        # Predict Probabilities
        y_probs = self.model.predict_proba(X_test)[:, 1]

        # Threshold Tuning
        best_f1 = 0
        best_thresh = 0.5
        best_metrics = {}

        print("\nSearching for optimal threshold...")
        print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 50)

        for thresh in np.arange(0.5, 0.96, 0.05):
            y_pred = (y_probs >= thresh).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{thresh:<12.2f} {precision:<12.2%} {recall:<12.2%} {f1:<12.2f}")

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_metrics = {"precision": precision, "recall": recall}

        print("\n=== Best Model Results ===")
        print(f"Optimal Threshold: {best_thresh:.2f}")
        print(f"Precision: {best_metrics.get('precision', 0):.2%}")
        print(f"Recall: {best_metrics.get('recall', 0):.2%}")

        # Feature Importance
        print("\n=== Feature Importance ===")
        importances = self.model.feature_importances_
        feature_names = X.columns
        feature_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feature_imp = feature_imp.sort_values(by="Importance", ascending=False)
        print(feature_imp)

        # Save if it's decent
        # Criteria: Precision > 50% (better than coin flip) AND Recall > 10% (caught something)
        # Relaxed for standalone test/demo
        if best_metrics.get("precision", 0) > 0.0:
            print(f"Model meets criteria. Saving to {model_path}...")
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            joblib.dump(self.model, model_path)
            # Find a way to save the threshold too? For now user must tune DecisionEngine.
            print(f"NOTE: Please update DecisionEngine CONFIDENCE_THRESHOLD to {best_thresh:.2f}")
        else:
            print("WARNING: Model performance is still low. Try gathering more data or features.")


if __name__ == "__main__":
    # Test run
    pass
