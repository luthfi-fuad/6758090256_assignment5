import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin

# Configure Plotting Style for Light Mode
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'axes.facecolor': '#ffffff',
    'figure.facecolor': '#ffffff',
    'axes.edgecolor': '#444444',
    'grid.color': '#cccccc',
    'legend.facecolor': '#ffffff',
    'legend.edgecolor': '#444444'
})

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TRAIN_DATA_PATH = 'data/train_data.csv'
MODEL_BUNDLE_PATH = 'examples/luffy_class_5_training_model_v2.joblib'
os.makedirs('examples', exist_ok=True)

# ==============================================================================
# MLP WRAPPER AND LOSS
# ==============================================================================
@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * cross_entropy)

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.model = None
        self.classes_ = np.array([0, 1])
        self.history_ = None

    def fit(self, X, y):
        self.model = keras.Sequential([
            keras.Input(shape=(X.shape[1],)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss=focal_loss, metrics=['accuracy'])

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',   # metric to watch
            min_delta=0.001,      # minimum change to qualify as an improvement
            patience=12,          # epochs to wait before stopping after no improvement
            restore_best_weights=True  # revert to best weights on stop
        )

        print("\nTraining MLP Component...")
        history = self.model.fit(
            X, y,
            epochs=100,
            batch_size=64,
            verbose=1,
            validation_split=0.15,
            callbacks=[early_stop]   # <-- pass it here
        )
        self.history_ = history.history
        return self

    def predict_proba(self, X):
        probs = self.model.predict(X, verbose=0)
        return np.hstack([1 - probs, probs])

    def predict(self, X):
        return (self.model.predict(X, verbose=0) > 0.5).astype(int)

# ==============================================================================
# CLEANING AND PLOTTING
# ==============================================================================
from feature_engineering import clean_data

def plot_training_history(history):
    if not history: return
    epochs = range(len(history['accuracy']))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(epochs, history['accuracy'], color='#00d9ff', label='Train Accuracy')
    if 'val_accuracy' in history: ax1.plot(epochs, history['val_accuracy'], color='#ff00ff', label='Val Accuracy')
    ax1.set_title("Clean Dataset MLP Training: Accuracy", fontweight='bold', pad=20)
    ax1.legend()
    
    ax2.plot(epochs, history['loss'], color='#00d9ff', label='Train Loss')
    if 'val_loss' in history: ax2.plot(epochs, history['val_loss'], color='#ff00ff', label='Val Loss')
    ax2.set_title("Clean Dataset MLP Training: Loss", fontweight='bold', pad=20)
    ax2.legend()
    plt.tight_layout()
    
    plt.savefig('output/luffy_class5_training_history_v2.png')
    plt.show()

def main():
    print(f"Initializing Clean Dataset Training on {TRAIN_DATA_PATH}...")
    df = pd.read_csv(TRAIN_DATA_PATH)
    df = clean_data(df)
    
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']
    
    cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # MANUAL OVERSAMPLING
    print("Applying Manual Oversampling to balance classes...")
    temp_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)
    temp_df['ProdTaken'] = y.values
    
    df_majority = temp_df[temp_df.ProdTaken == 0]
    df_minority = temp_df[temp_df.ProdTaken == 1]
    
    df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True, random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    X_balanced = df_balanced.drop('ProdTaken', axis=1)
    y_balanced = df_balanced['ProdTaken']
    
    print(f"Original samples: {len(df)}, Balanced samples: {len(df_balanced)}")
    
    print("\nTraining Deep Stacked Ensemble (HistGB, RF, ET, MLP)...")
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=1000, max_depth=25, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=1000, max_depth=25, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=10, random_state=42)),
        ('hgb', HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_depth=12, random_state=42)),
        ('mlp', KerasClassifierWrapper())
    ]
    
    stack = StackingClassifier(
        estimators=estimators, 
        final_estimator=RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42),
        cv=5, 
        n_jobs=-1
    )
    stack.fit(X_balanced, y_balanced)
    
    print("\nVisualizing MLP Training Evolution...")
    plot_training_history(stack.named_estimators_['mlp'].history_)
    
    print(f"Saving clean model bundle to {MODEL_BUNDLE_PATH}")
    joblib.dump({'model': stack, 'scaler': scaler, 'columns': X_encoded.columns.tolist()}, MODEL_BUNDLE_PATH)
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()