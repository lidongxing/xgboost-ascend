import numpy as np
import pandas as pd
from xgboost_npu import XGBoostNPU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import pickle
import os
warnings.filterwarnings('ignore')

# 设置中文字体，使用arm64 Ubuntu系统中可用的字体
from matplotlib.font_manager import FontProperties

possible_font_paths = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]

font_path = None
for path in possible_font_paths:
    if os.path.exists(path):
        font_path = path
        break

if font_path:
    my_font = FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [my_font.get_name()]
    print(f"使用字体: {my_font.get_name()} ({font_path})")
else:
    print("未找到可用中文字体，中文可能无法正常显示。")
plt.rcParams['axes.unicode_minus'] = False



print("=" * 60)
print("信用卡违约预测 - 二分类模型")
print("=" * 60)

# 1. 数据加载和基础分析
print("1. 数据加载和基础分析...")
# 跳过前两行并从第二列开始加载数据
data = pd.read_csv('default of credit card clients.csv', skiprows=2, usecols=lambda x: x != 0)

print(f"原始数据形状: {data.shape}")
print(f"列名: {list(data.columns)}")

# 检查数据基本信息
print(f"\n数据类型:")
print(data.dtypes)

print(f"\n缺失值统计:")
print(data.isnull().sum())

print(f"\n前5行数据:")
print(data.head())

# 2. 数据预处理
print("\n2. 数据预处理...")

# 假设最后一列是目标变量，检查目标变量
target_col = data.columns[-1]
print(f"目标变量: {target_col}")
print(f"目标变量原始值分布:")
print(data[target_col].value_counts())

# 转换目标变量为数值型
if data[target_col].dtype == 'object':  # 如果是字符串类型
    # 先查看唯一值，确定转换逻辑
    unique_values = data[target_col].unique()
    print(f"目标变量唯一值: {unique_values}")
    
    # 创建一个函数来判断是否可以转换为整数
    def is_integer(s):
        try:
            int(s)
            return True
        except ValueError:
            return False
    
    # 识别所有不能转换为整数的值
    invalid_values = [v for v in unique_values if not is_integer(v)]
    
    if invalid_values:
        print(f"警告: 发现非整数值 {invalid_values}，将其从数据中过滤")
        # 过滤掉所有不能转换为整数的行
        valid_mask = data[target_col].apply(is_integer)
        data = data[valid_mask].copy()
        print(f"过滤后数据形状: {data.shape}")
        print(f"过滤后目标变量唯一值: {data[target_col].unique()}")
    
    # 转换为数值类型
    data[target_col] = data[target_col].astype(int)
    
print(f"目标变量转换后分布:")
print(data[target_col].value_counts())
print(f"违约率: {data[target_col].mean():.4f}")

# 分离特征和目标
X = data.iloc[:, :-1].values  # 所有列除了最后一列
y = data.iloc[:, -1].values   # 最后一列作为目标

print(f"\n特征形状: X={X.shape}")
print(f"目标形状: y={y.shape}")

# 检查特征类型
print(f"\n特征统计:")
print(f"  数值特征数: {X.shape[1]}")
print(f"  样本数: {X.shape[0]}")
print(f"  正样本数: {np.sum(y == 1)}")
print(f"  负样本数: {np.sum(y == 0)}")
print(f"  正负样本比例: {np.sum(y == 1) / np.sum(y == 0):.3f}")

# 3. 特征工程
print("\n3. 特征工程...")

# 使用RobustScaler处理异常值
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 添加多项式特征（可选，用于提高模型性能）
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X_scaled)

print(f"原始特征数: {X.shape[1]}")
print(f"多项式特征数: {X_poly.shape[1]}")

# 4. 数据划分
print("\n4. 数据划分...")
# 使用分层抽样确保训练集和测试集中的类别比例一致
X_temp, X_test, y_temp, y_test = train_test_split(
    X_poly, y, test_size=0.1, random_state=42, shuffle=True, stratify=y
)

X_train, X_eval, y_train, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, shuffle=True, stratify=y_temp
)

print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征 (80%)")
print(f"验证集: {X_eval.shape[0]} 样本, {X_eval.shape[1]} 特征 (10%)")
print(f"测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征 (10%)")

print(f"\n类别分布:")
print(f"  训练集 - 正样本: {np.sum(y_train == 1)}, 负样本: {np.sum(y_train == 0)}")
print(f"  验证集 - 正样本: {np.sum(y_eval == 1)}, 负样本: {np.sum(y_eval == 0)}")
print(f"  测试集 - 正样本: {np.sum(y_test == 1)}, 负样本: {np.sum(y_test == 0)}")

# 5. 模型训练
print("\n5. 模型训练...")
model = XGBoostNPU(
    max_depth=6,           # 树深度
    min_samples_split=10,  # 分裂阈值
    learning_rate=0.1,     # 学习率
    n_estimators=100,      # 树的数量
    lambda_param=0.1,      # L2正则化
    task_type='binary'     # 二分类任务
)

print("开始训练...")
train_start_time = time.time()
model.train(X_train, y_train)
train_end_time = time.time()
train_time = train_end_time - train_start_time
print(f"训练完成! 训练时间: {train_time:.2f} 秒")

# 保存模型
print("\n保存模型...")
model_save_path = "credit_card_model.pkl"
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ 模型已保存到: {model_save_path}")
except Exception as e:
    print(f"⚠ 模型保存失败: {e}")

# 6. 预测和评估
print("\n6. 预测和评估...")

# 训练集预测
print("训练集预测...")
train_start_time = time.time()
train_preds_proba = model.predict(X_train)
train_preds = (train_preds_proba > 0.5).astype(int)
train_end_time = time.time()
train_pred_time = train_end_time - train_start_time
print(f"训练集预测完成! 预测时间: {train_pred_time:.4f} 秒")

# 验证集预测
print("验证集预测...")
eval_start_time = time.time()
eval_preds_proba = model.predict(X_eval)
eval_preds = (eval_preds_proba > 0.5).astype(int)
eval_end_time = time.time()
eval_pred_time = eval_end_time - eval_start_time
print(f"验证集预测完成! 预测时间: {eval_pred_time:.4f} 秒")

# 测试集预测
print("测试集预测...")
test_start_time = time.time()
test_preds_proba = model.predict(X_test)
test_preds = (test_preds_proba > 0.5).astype(int)
test_end_time = time.time()
test_pred_time = test_end_time - test_start_time
print(f"测试集预测完成! 预测时间: {test_pred_time:.4f} 秒")

# 7. 评估指标
print("\n7. 评估指标...")

def evaluate_classification(y_true, y_pred, y_proba, dataset_name, pred_time=None):
    """评估分类性能"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"{dataset_name}结果:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    if pred_time is not None:
        print(f"  预测时间: {pred_time:.4f} 秒")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_proba,
        'prediction_time': pred_time
    }

# 评估各个数据集
train_results = evaluate_classification(y_train, train_preds, train_preds_proba, "训练集", train_pred_time)
eval_results = evaluate_classification(y_eval, eval_preds, eval_preds_proba, "验证集", eval_pred_time)
test_results = evaluate_classification(y_test, test_preds, test_preds_proba, "测试集", test_pred_time)

# 8. 混淆矩阵分析
print("\n8. 混淆矩阵分析...")

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['非违约', '违约'], 
                yticklabels=['非违约', '违约'])
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算详细指标
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"  {title}:")
    print(f"    真阴性 (TN): {tn}")
    print(f"    假阳性 (FP): {fp}")
    print(f"    假阴性 (FN): {fn}")
    print(f"    真阳性 (TP): {tp}")
    print(f"    特异度 (Specificity): {specificity:.4f}")
    print(f"    敏感度 (Sensitivity): {sensitivity:.4f}")

plot_confusion_matrix(y_test, test_preds, "测试集混淆矩阵", "test_confusion_matrix.png")
plot_confusion_matrix(y_train, train_preds, "训练集混淆矩阵", "train_confusion_matrix.png")

# 9. 保存结果
print("\n9. 保存结果...")

# 准备原始数据（用于保存到文件）
original_data = pd.read_csv('default of credit card clients.csv', skiprows=2, usecols=lambda x: x != 0)

# 获取原始数据的索引
all_indices = np.arange(len(original_data))
_, test_indices = train_test_split(all_indices, test_size=0.1, random_state=42, shuffle=True, stratify=y)
temp_indices = np.setdiff1d(all_indices, test_indices)
train_indices, eval_indices = train_test_split(temp_indices, test_size=0.125, random_state=42, shuffle=True, stratify=y[temp_indices])

# 创建训练文件
train_data = original_data.iloc[train_indices].copy()
train_data.to_csv('credit_train.csv', index=False)
print("✓ 训练文件已保存: credit_train.csv")

# 创建验证文件
eval_data = original_data.iloc[eval_indices].copy()
eval_data.to_csv('credit_eval.csv', index=False)
print("✓ 验证文件已保存: credit_eval.csv")

# 创建测试文件（包含预测列）
test_data = original_data.iloc[test_indices].copy()
test_data['default_pred'] = test_preds  # 添加预测列
test_data['default_probability'] = test_preds_proba  # 添加概率列
test_data.to_csv('credit_test.csv', index=False)
print("✓ 测试文件已保存: credit_test.csv (包含预测列)")

# 保存详细结果
results_df = pd.DataFrame({
    '真实标签': y_test,
    '预测标签': test_preds,
    '预测概率': test_preds_proba,
    '预测正确': (y_test == test_preds)
})

results_df.to_csv('credit_classification_results.csv', index=False)
print("✓ 详细结果已保存到 credit_classification_results.csv")

# 保存时间性能信息
time_performance = {
    '训练时间(秒)': train_time,
    '训练集样本数': len(X_train),
    '训练集预测时间(秒)': train_pred_time,
    '验证集样本数': len(X_eval),
    '验证集预测时间(秒)': eval_pred_time,
    '测试集样本数': len(X_test),
    '测试集预测时间(秒)': test_pred_time,
    '总训练时间(秒)': train_time,
    '总预测时间(秒)': train_pred_time + eval_pred_time + test_pred_time,
    '总样本数': len(X_train) + len(X_eval) + len(X_test)
}

time_df = pd.DataFrame([time_performance])
time_df.to_csv('credit_time_performance.csv', index=False)
print("✓ 时间性能信息已保存到 credit_time_performance.csv")

# 显示文件信息
print(f"\n文件信息:")
print(f"  credit_train.csv: {len(train_data)} 样本 (训练用)")
print(f"  credit_eval.csv: {len(eval_data)} 样本 (验证用)")
print(f"  credit_test.csv: {len(test_data)} 样本 (测试用，包含预测列)")
print(f"  credit_card_model.pkl: 训练好的模型文件")
print(f"  credit_time_performance.csv: 时间性能信息")

# 10. 可视化
print("\n10. 生成可视化图表...")
plt.figure(figsize=(15, 10))

# 子图1: ROC曲线
plt.subplot(2, 3, 1)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, test_preds_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {test_results["auc"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend()

# 子图2: 预测概率分布
plt.subplot(2, 3, 2)
plt.hist(test_preds_proba[y_test == 0], bins=30, alpha=0.7, label='非违约', edgecolor='black')
plt.hist(test_preds_proba[y_test == 1], bins=30, alpha=0.7, label='违约', edgecolor='black')
plt.xlabel('预测概率')
plt.ylabel('频数')
plt.title('预测概率分布')
plt.legend()

# 子图3: 性能指标对比
plt.subplot(2, 3, 3)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
train_values = [train_results['accuracy'], train_results['precision'], 
                train_results['recall'], train_results['f1'], train_results['auc']]
test_values = [test_results['accuracy'], test_results['precision'], 
               test_results['recall'], test_results['f1'], test_results['auc']]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_values, width, label='训练集', alpha=0.8)
plt.bar(x + width/2, test_values, width, label='测试集', alpha=0.8)
plt.xlabel('指标')
plt.ylabel('分数')
plt.title('性能指标对比')
plt.xticks(x, metrics, rotation=45)
plt.legend()

# 子图4: 类别分布
plt.subplot(2, 3, 4)
train_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
test_counts = [np.sum(y_test == 0), np.sum(y_test == 1)]

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, train_counts, width, label='训练集', alpha=0.8)
plt.bar(x + width/2, test_counts, width, label='测试集', alpha=0.8)
plt.xlabel('类别')
plt.ylabel('样本数')
plt.title('类别分布')
plt.xticks(x, ['非违约', '违约'])
plt.legend()

# 子图5: 预测概率箱线图
plt.subplot(2, 3, 5)
plt.boxplot([test_preds_proba[y_test == 0], test_preds_proba[y_test == 1]], 
            labels=['非违约', '违约'])
plt.ylabel('预测概率')
plt.title('预测概率箱线图')

# 子图6: 混淆矩阵热图
plt.subplot(2, 3, 6)
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['非违约', '违约'], 
            yticklabels=['非违约', '违约'])
plt.title('测试集混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')

plt.tight_layout()
plt.savefig('credit_classification_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 可视化图表已保存到 credit_classification_results.png")

# 11. 分类报告
print("\n11. 详细分类报告...")
print("测试集分类报告:")
print(classification_report(y_test, test_preds, target_names=['非违约', '违约']))

# 12. 总结
print("\n" + "="*60)
print("信用卡违约预测总结")
print("="*60)
print("主要特点:")
print("1. 二分类任务 - 预测信用卡违约")
print("2. 数据不平衡 - 违约样本较少")
print("3. 多项式特征 - 增加特征交互")
print("4. 分层抽样 - 保持类别比例")
print("5. 多指标评估 - 全面性能分析")
print("6. 模型保存 - 可重复使用训练好的模型")
print("7. 时间记录 - 记录训练和预测时间")

print(f"\n时间性能:")
print(f"训练时间: {train_time:.2f} 秒")
print(f"训练集预测时间: {train_pred_time:.4f} 秒 (样本数: {len(X_train)})")
print(f"验证集预测时间: {eval_pred_time:.4f} 秒 (样本数: {len(X_eval)})")
print(f"测试集预测时间: {test_pred_time:.4f} 秒 (样本数: {len(X_test)})")
print(f"总预测时间: {train_pred_time + eval_pred_time + test_pred_time:.4f} 秒 (总样本数: {len(X_train) + len(X_eval) + len(X_test)})")

print(f"\n最终性能:")
print(f"训练集 - 准确率: {train_results['accuracy']:.4f}, AUC: {train_results['auc']:.4f}")
print(f"验证集 - 准确率: {eval_results['accuracy']:.4f}, AUC: {eval_results['auc']:.4f}")
print(f"测试集 - 准确率: {test_results['accuracy']:.4f}, AUC: {test_results['auc']:.4f}")

print(f"\n输出文件:")
print(f"  credit_train.csv - 训练数据 (80%)")
print(f"  credit_eval.csv - 验证数据 (10%)")
print(f"  credit_test.csv - 测试数据 (10%) + 预测列")
print(f"  credit_card_model.pkl - 训练好的模型")
print(f"  credit_time_performance.csv - 时间性能信息")
print(f"  credit_classification_results.png - 可视化图表")
print(f"  credit_classification_results.csv - 详细结果")

if test_results['auc'] > 0.8:
    print("\n✓ 性能优秀!")
elif test_results['auc'] > 0.7:
    print("\n✓ 性能良好!")
elif test_results['auc'] > 0.6:
    print("\n✓ 性能可接受!")
else:
    print("\n⚠ 性能需要进一步改进")

print("\n信用卡违约预测完成!")