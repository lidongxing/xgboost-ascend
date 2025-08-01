import ctypes
import numpy as np
from mindspore import Tensor
import os
import matplotlib.pyplot as plt
from enum import Enum

class TaskType(Enum):
    BINARY_CLASSIFICATION = 'binary'
    MULTICLASS_CLASSIFICATION = 'multiclass'
    REGRESSION = 'regression'

class XGBoostNPU:
    def __init__(self, max_depth=4, min_samples_split=5, learning_rate=0.3, 
                 n_estimators=50, lambda_param=0.1, task_type='binary', num_class=2):
        """
        初始化XGBoost模型
        @param task_type: 任务类型 ('binary', 'multiclass', 'regression')
        @param num_class: 多分类任务的类别数量
        """
        self.task_type = TaskType(task_type)
        self.num_class = num_class if self.task_type == TaskType.MULTICLASS_CLASSIFICATION else 1
        
        # 基本参数
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.lambda_param = lambda_param
        
        # 加载动态库
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libxgboost_npu.so')
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Could not find libxgboost_npu.so at {lib_path}")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # 设置函数签名
        self._setup_lib_functions()
        
        # 创建模型配置
        config = self._create_config()
        self.model = self.lib.create_xgboost_model(config)
        if not self.model:
            raise RuntimeError("Failed to create model")

    def _setup_lib_functions(self):
        """设置C库函数的签名"""
        class XGBoostConfig(ctypes.Structure):
            _fields_ = [
                ("max_depth", ctypes.c_int),
                ("min_samples_split", ctypes.c_int),
                ("learning_rate", ctypes.c_float),
                ("n_estimators", ctypes.c_int),
                ("lambda_param", ctypes.c_float),
                ("task_type", ctypes.c_int),  # 0: binary, 1: multiclass, 2: regression
                ("num_class", ctypes.c_int)
            ]
        self.XGBoostConfig = XGBoostConfig
        
        # 设置函数签名
        self.lib.create_xgboost_model.argtypes = [XGBoostConfig]
        self.lib.create_xgboost_model.restype = ctypes.c_void_p

        self.lib.train.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int
        ]
        self.lib.train.restype = ctypes.c_int

        self.lib.predict.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int
        ]
        self.lib.predict.restype = ctypes.POINTER(ctypes.c_float)

        self.lib.free_xgboost_model.argtypes = [ctypes.c_void_p]
        print("Function types set successfully")

    def _create_config(self):
        """创建模型配置"""
        task_type_map = {
            TaskType.BINARY_CLASSIFICATION: 0,
            TaskType.MULTICLASS_CLASSIFICATION: 1,
            TaskType.REGRESSION: 2
        }
        
        return self.XGBoostConfig(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            lambda_param=self.lambda_param,
            task_type=task_type_map[self.task_type],
            num_class=self.num_class
        )

    def train(self, X, y):
        """
        训练模型
        @param X: 特征矩阵
        @param y: 目标变量 (对于多分类，应该是1D标签数组，不是one-hot编码)
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        
        # 确保y是1D数组，对于多分类任务不要进行one-hot编码
        if len(y.shape) > 1:
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                # 如果传入的是one-hot编码，转换为1D标签
                y = np.argmax(y, axis=1).astype(np.float32)
            else:
                # 对于其他任务，展平数组
                y = y.flatten().astype(np.float32)
        
        print(f"\nTraining {self.task_type.value} model:")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"y unique values: {np.unique(y)}")
        print(f"y value counts: {np.bincount(y.astype(int))}")
        
        result = self.lib.train(self.model, 
                              X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              X.shape[0],
                              X.shape[1])
        
        if result != 0:
            raise RuntimeError("Training failed")

    def predict(self, X):
        """
        预测样本的类别概率
        @param X: 特征矩阵
        @return: 预测概率
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        n_samples, n_features = X.shape
        
        # 分配预测结果内存
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            predictions = np.zeros((n_samples, self.num_class), dtype=np.float32)
        else:
            predictions = np.zeros(n_samples, dtype=np.float32)
        
        # 调用C函数进行预测
        predictions_ptr = self.lib.predict(
            self.model,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_samples),
            ctypes.c_int(n_features)
        )
        
        # 将C返回的预测结果复制到numpy数组
        predictions_size = n_samples * (self.num_class if self.task_type == TaskType.MULTICLASS_CLASSIFICATION else 1)
        ctypes.memmove(predictions.ctypes.data, predictions_ptr, predictions_size * ctypes.sizeof(ctypes.c_float))
        
        # 释放C分配的内存
        self.lib.free(predictions_ptr)
        
        return predictions

    def evaluate(self, X, y):
        """
        评估模型性能
        @param X: 特征矩阵
        @param y: 真实标签 (对于多分类，应该是1D标签数组)
        """
        # 确保y是1D数组
        if len(y.shape) > 1:
            if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                y = np.argmax(y, axis=1)
            else:
                y = y.flatten()
        
        predictions = self.predict(X)
        
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            # 二分类评估
            pred_labels = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_labels == y)
            
            # 计算TP, FP, TN, FN
            TP = np.sum((pred_labels == 1) & (y == 1))
            FP = np.sum((pred_labels == 1) & (y == 0))
            TN = np.sum((pred_labels == 0) & (y == 0))
            FN = np.sum((pred_labels == 0) & (y == 1))
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print("\n评估指标:")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}\n")
            
            # 分析预测分布
            print("预测分布分析:")
            print(f"平均预测值: {predictions.mean():.4f}")
            print(f"预测标准差: {predictions.std():.4f}")
            print(f"预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]\n")
            
            # 分析类别准确率
            for label in [0, 1]:
                mask = y == label
                if mask.any():
                    class_acc = np.mean(pred_labels[mask] == y[mask])
                    print(f"类别 {label} 准确率: {class_acc:.4f}")
                    print(f"类别 {label} 样本数: {mask.sum()}")
            
            # 分析预测置信度
            confidences = np.maximum(predictions, 1 - predictions)
            print("\n预测置信度分析:")
            print(f"平均置信度: {confidences.mean():.4f}")
            print("置信度分布:")
            for threshold in [0.2, 0.4, 0.6, 0.8]:
                ratio = (confidences >= threshold).mean() * 100
                print(f"置信度 >= {threshold*100:.1f}% 的预测比例: {ratio:.1f}%")
            
            # 打印一些预测示例
            print("\n预测示例 (前10个样本):")
            for i in range(min(10, len(y))):
                correct = "✓" if pred_labels[i] == y[i] else "✗"
                confidence = confidences[i] * 100
                print(f"样本 {i}: 真实值={y[i]}, 预测概率={predictions[i]:.4f}, "
                      f"预测类别={pred_labels[i]}, 置信度={confidence:.1f}% {correct}")
        
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # 多分类评估
            pred_labels = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_labels == y)
            
            print("\n多分类评估指标:")
            print(f"准确率: {accuracy:.4f}")
            
            # 分析每个类别的性能
            print("\n各类别分析:")
            for class_idx in range(self.num_class):
                mask = y == class_idx
                if mask.any():
                    class_acc = np.mean(pred_labels[mask] == y[mask])
                    class_count = mask.sum()
                    print(f"类别 {class_idx}: 准确率={class_acc:.4f}, 样本数={class_count}")
            
            # 分析预测分布
            print("\n预测分布分析:")
            print(f"预测类别分布: {np.bincount(pred_labels, minlength=self.num_class)}")
            print(f"真实类别分布: {np.bincount(y.astype(int), minlength=self.num_class)}")
            
            # 分析预测概率分布
            for class_idx in range(self.num_class):
                class_probs = predictions[:, class_idx]
                print(f"类别 {class_idx} 预测概率: 平均={class_probs.mean():.4f}, "
                      f"标准差={class_probs.std():.4f}, 范围=[{class_probs.min():.4f}, {class_probs.max():.4f}]")
            
            # 打印一些预测示例
            print("\n预测示例 (前10个样本):")
            for i in range(min(10, len(y))):
                correct = "✓" if pred_labels[i] == y[i] else "✗"
                max_prob = predictions[i].max()
                print(f"样本 {i}: 真实值={y[i]}, 预测类别={pred_labels[i]}, "
                      f"最大概率={max_prob:.4f} {correct}")
                print(f"  各类别概率: {predictions[i]}")
        
        else:
            # 回归评估
            mse = np.mean((predictions - y) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y))
            
            print("\n回归评估指标:")
            print(f"均方误差 (MSE): {mse:.4f}")
            print(f"均方根误差 (RMSE): {rmse:.4f}")
            print(f"平均绝对误差 (MAE): {mae:.4f}")
            
            # 分析预测分布
            print("\n预测分布分析:")
            print(f"真实值范围: [{y.min():.4f}, {y.max():.4f}]")
            print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
            print(f"真实值平均: {y.mean():.4f}")
            print(f"预测值平均: {predictions.mean():.4f}")

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'model') and self.lib and self.model:
            try:
                self.lib.free_xgboost_model(self.model)
                print("Model resources freed successfully")
            except Exception as e:
                print(f"Error freeing resources: {e}")
