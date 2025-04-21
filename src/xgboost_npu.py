import ctypes
import numpy as np
import mindspore as ms
from mindspore import Tensor
import os

class XGBoostConfig(ctypes.Structure):
    _fields_ = [
        ("max_depth", ctypes.c_int),
        ("min_samples_split", ctypes.c_int),
        ("learning_rate", ctypes.c_float),
        ("n_estimators", ctypes.c_int)
    ]

class XGBoostNPU:
    def __init__(self, max_depth=2, min_samples_split=2, learning_rate=0.1, n_estimators=100):
        print("Initializing XGBoostNPU...")
        
        # 获取动态库的绝对路径
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libxgboost_npu.so')
        print(f"Looking for library at: {lib_path}")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Could not find libxgboost_npu.so at {lib_path}")
        
        # 加载动态库
        try:
            self.lib = ctypes.CDLL(lib_path)
            print("Library loaded successfully")
        except Exception as e:
            print(f"Failed to load library: {e}")
            raise
        
        # 设置函数参数和返回类型
        try:
            self.lib.create_xgboost_model.argtypes = [XGBoostConfig]
            self.lib.create_xgboost_model.restype = ctypes.c_void_p
            
            self.lib.train_xgboost_model.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int
            ]
            
            self.lib.predict_xgboost_model.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int
            ]
            self.lib.predict_xgboost_model.restype = ctypes.c_float
            
            self.lib.free_xgboost_model.argtypes = [ctypes.c_void_p]
            print("Function types set successfully")
        except Exception as e:
            print(f"Failed to set function types: {e}")
            raise
        
        # 创建配置
        print(f"Creating config: max_depth={max_depth}, min_samples_split={min_samples_split}, "
              f"learning_rate={learning_rate}, n_estimators={n_estimators}")
        config = XGBoostConfig(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            learning_rate=learning_rate,
            n_estimators=n_estimators
        )
        
        # 创建模型
        try:
            self.model = self.lib.create_xgboost_model(config)
            if not self.model:
                raise RuntimeError("Failed to create XGBoost model")
            print("Model created successfully")
        except Exception as e:
            print(f"Failed to create model: {e}")
            raise

    def train(self, X, y):
        print("Starting training...")
        # 确保输入是 MindSpore Tensor
        if isinstance(X, Tensor):
            X = X.asnumpy()
        if isinstance(y, Tensor):
            y = y.asnumpy()
            
        # 转换为 float32
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # 调用训练函数
        try:
            result = self.lib.train_xgboost_model(
                self.model,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                X.shape[0],
                X.shape[1]
            )
            if result != 0:
                raise RuntimeError("Failed to train XGBoost model")
            print("Training completed successfully")
        except Exception as e:
            print(f"Training failed: {e}")
            raise
    
    def predict(self, X):
        """Make predictions using the trained model"""
        print("Starting prediction in Python...")
        if not self.model:
            raise ValueError("Model not trained")
            
        # 确保输入是numpy数组
        if isinstance(X, Tensor):
            X = X.asnumpy()
        elif not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
            
        print(f"Input data shape: {X.shape}")
        print(f"Input data type: {X.dtype}")
        
        n_samples = X.shape[0]
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        
        print(f"Number of samples: {n_samples}")
        print(f"Number of features: {n_features}")
        
        # 确保数据是连续的并且是float32类型
        X = np.ascontiguousarray(X, dtype=np.float32)
        
        # 获取预测函数
        predict_func = self.lib.predict_xgboost_model
        predict_func.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        predict_func.restype = ctypes.POINTER(ctypes.c_float)
        
        print("Calling C prediction function...")
        try:
            # 将numpy数组转换为C指针
            X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            print(f"Input data pointer: {X_ptr}")
            print(f"Model pointer: {self.model}")
            
            predictions_ptr = predict_func(self.model, X_ptr, n_samples)
            print(f"Predictions pointer: {predictions_ptr}")
            
            if not predictions_ptr:
                raise RuntimeError("Prediction failed")
                
            # 将预测结果转换为numpy数组
            predictions = np.ctypeslib.as_array(predictions_ptr, shape=(n_samples,))
            print(f"Predictions shape: {predictions.shape}")
            
            # 检查并处理NaN值和数值范围
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                print("Warning: NaN or infinite values detected in predictions")
                # 用0.5替换无效值
                predictions = np.nan_to_num(predictions, nan=0.5, posinf=0.5, neginf=0.5)
            
            # 确保预测值在[0,1]范围内
            predictions = np.clip(predictions, 0.0, 1.0)
            
            print(f"First few predictions: {predictions[:5]}")
            
            # 确保数据类型是float32
            predictions = predictions.astype(np.float32)
            
            # 直接返回numpy数组
            return predictions
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def __del__(self):
        if hasattr(self, 'model') and self.model:
            try:
                self.lib.free_xgboost_model(self.model)
                print("Model resources freed successfully")
            except Exception as e:
                print(f"Failed to free model resources: {e}")
