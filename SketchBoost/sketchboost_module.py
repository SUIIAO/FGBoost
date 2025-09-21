import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Add the Py-Boost path to sys.path
py_boost_path = "/Users/suiiao/Google Drive/XGBoost/NeurIPS_submission_rebuttal/Py-Boost-master"
if py_boost_path not in sys.path:
    sys.path.insert(0, py_boost_path)

# Try to import the real SketchBoost from Py-Boost
try:
    from py_boost.cpu_sketch_boost import CPUSketchBoost as PyBoostSketchBoost
    PYBOOST_AVAILABLE = True
    print("Successfully imported CPU SketchBoost from Py-Boost")
except ImportError as e:
    print(f"Warning: Could not import Py-Boost CPU SketchBoost: {e}")
    PYBOOST_AVAILABLE = False
except Exception as e:
    print(f"Warning: Error importing Py-Boost CPU SketchBoost: {e}")
    PYBOOST_AVAILABLE = False

# Fallback: Import XGBoost for CPU implementation if Py-Boost fails
if not PYBOOST_AVAILABLE:
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
        raise ImportError("Neither Py-Boost nor XGBoost is available")

def sketch_outputs(y, sketch_dim=5, method='random'):
    """
    Apply sketching to reduce the dimensionality of multi-output targets.
    
    Args:
        y: Target matrix of shape (n_samples, n_outputs)
        sketch_dim: Desired sketch dimension
        method: Sketching method ('random' supported)
    
    Returns:
        y_sketch: Sketched targets of shape (n_samples, sketch_dim)
        sketch_matrix: The sketching transformation matrix
    """
    n_samples, n_outputs = y.shape
    
    if method == 'random':
        # Random projection sketching
        np.random.seed(42)  # For reproducibility
        sketch_matrix = np.random.randn(n_outputs, sketch_dim) / np.sqrt(sketch_dim)
        y_sketch = y @ sketch_matrix
    else:
        raise ValueError(f"Unsupported sketching method: {method}")
    
    return y_sketch, sketch_matrix

def reconstruct_outputs(y_sketch, sketch_matrix):
    """
    Reconstruct full outputs from sketched predictions.
    
    Args:
        y_sketch: Sketched predictions of shape (n_samples, sketch_dim)
        sketch_matrix: The sketching transformation matrix of shape (n_outputs, sketch_dim)
    
    Returns:
        y_reconstructed: Reconstructed outputs of shape (n_samples, n_outputs)
    """
    reconstruction_matrix = np.linalg.pinv(sketch_matrix)
    y_reconstructed = y_sketch @ reconstruction_matrix
    return y_reconstructed

class SketchBoost:
    """
    Wrapper for SketchBoost implementation - tries Py-Boost first, falls back to XGBoost.
    """
    
    def __init__(self, sketch_dim=5, sketch_method='random', n_estimators=100, 
                 max_depth=4, learning_rate=0.1, random_state=42, force_fallback=False):
        """
        Initialize SketchBoost model.
        
        Args:
            sketch_dim: Dimension for sketching
            sketch_method: Sketching method ('random', 'proj', 'topk')
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
            force_fallback: Force use of XGBoost fallback implementation
        """
        self.sketch_dim = sketch_dim
        self.sketch_method = sketch_method
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.is_fitted = False
        self.use_pyboost = PYBOOST_AVAILABLE and not force_fallback
        
        if self.use_pyboost:
            # Map sketch methods to Py-Boost CPU format
            sketch_method_map = {
                'random': 'rand',
                'proj': 'proj', 
                'topk': 'topk',
                'best': 'topk'
            }
            
            py_boost_method = sketch_method_map.get(sketch_method, 'proj')
            
            # Initialize the CPU Py-Boost SketchBoost
            try:
                self.model = PyBoostSketchBoost(
                    loss='mse',
                    metric='rmse',
                    ntrees=n_estimators,
                    lr=learning_rate,
                    max_depth=max_depth,
                    sketch_outputs=sketch_dim,
                    sketch_method=py_boost_method,
                    seed=random_state,
                    verbose=1 if sketch_dim <= 5 else 0  # Reduce verbosity for large sketches
                )
                print("Using Py-Boost CPU SketchBoost implementation")
            except Exception as e:
                print(f"Failed to initialize Py-Boost CPU SketchBoost: {e}")
                print("Falling back to XGBoost implementation")
                self.use_pyboost = False
        
        if not self.use_pyboost:
            # Fallback: Use XGBoost with manual sketching
            self.models = []
            self.sketch_matrix = None
            print("Using XGBoost fallback implementation with manual sketching")
        
    def fit(self, X, y):
        """
        Fit SketchBoost model.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples, n_outputs)
        
        Returns:
            self: Fitted model
        """
        # Convert to numpy if needed
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        if self.use_pyboost:
            # Try Py-Boost CPU implementation
            try:
                # CPU SketchBoost supports eval_sets parameter
                eval_sets = None  # Can be extended to support validation
                self.model.fit(X, y, eval_sets=eval_sets)
                self.is_fitted = True
                print("Successfully fitted using Py-Boost CPU SketchBoost")
            except Exception as e:
                # If CPU version fails, fall back to XGBoost
                print(f"Py-Boost CPU SketchBoost failed: {e}")
                print("Falling back to XGBoost implementation")
                self.use_pyboost = False
                self.models = []
                self.sketch_matrix = None
        
        if not self.use_pyboost:
            # Fallback XGBoost implementation
            if not XGBOOST_AVAILABLE:
                raise RuntimeError("XGBoost fallback is not available")
                
            # Apply sketching to targets
            y_sketch, self.sketch_matrix = sketch_outputs(y, self.sketch_dim, self.sketch_method)
            
            # Train separate XGBoost models for each sketched output
            self.models = []
            for i in range(y_sketch.shape[1]):
                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state
                )
                model.fit(X, y_sketch[:, i])
                self.models.append(model)
                
            self.is_fitted = True
            print("Successfully fitted using XGBoost fallback")
                
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Test features of shape (n_samples, n_features)
        
        Returns:
            y_pred: Predictions of shape (n_samples, n_outputs)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Convert to numpy if needed
        X = np.array(X, dtype=np.float32)
        
        if self.use_pyboost:
            # Use Py-Boost prediction
            y_pred = self.model.predict(X)
        else:
            # Use XGBoost fallback prediction
            # Predict sketched outputs
            y_sketch_pred = np.column_stack([model.predict(X) for model in self.models])
            
            # Reconstruct full outputs
            y_pred = reconstruct_outputs(y_sketch_pred, self.sketch_matrix)
        
        return y_pred
    
    def score(self, X, y):
        """
        Calculate R² score on test data.
        
        Args:
            X: Test features
            y: True targets
        
        Returns:
            r2: R² score
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

# Main function for R interface
def fit_sketchboost(X_train, y_train, X_test=None, y_test=None, 
                   sketch_dim=5, n_estimators=100, max_depth=4, 
                   learning_rate=0.1, random_state=42):
    """
    Fit SketchBoost model and optionally evaluate on test set.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features (optional)
        y_test: Test targets (optional)
        sketch_dim: Sketching dimension
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        random_state: Random seed
    
    Returns:
        Dictionary with model and results
    """
    # Fit the model
    model = SketchBoost(
        sketch_dim=sketch_dim,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    # Prepare results
    results = {
        'model': model,
        'sketch_dim': sketch_dim,
        'original_dim': y_train.shape[1] if hasattr(y_train, 'shape') else len(y_train[0])
    }
    
    # Add test results if test data provided
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        results['test_predictions'] = y_pred
        results['test_mse'] = mean_squared_error(y_test, y_pred)
        results['test_r2'] = r2_score(y_test, y_pred)
    
    return results

def predict_sketchboost(model, X):
    """
    Make predictions with a fitted SketchBoost model.
    
    Args:
        model: Fitted SketchBoost model
        X: Features to predict on
    
    Returns:
        Predictions
    """
    return model.predict(X) 