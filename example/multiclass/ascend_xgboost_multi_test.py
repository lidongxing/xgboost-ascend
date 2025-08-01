import numpy as np
import pandas as pd
from xgboost_npu import XGBoostNPU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import pickle
import os
warnings.filterwarnings('ignore')

# 设置输出目录
OUTPUT_DIR = "/home/HwHiAiUser/Desktop/2t/multiclass/test"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文字体，自动检测可用字体
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
print("多分类任务 - XGBoostNPU示例")
print("=" * 60)

# 1. 数据加载
print("1. 从CSV文件加载数据...")
data_path = "test.csv"

# 检查文件是否存在
if not os.path.exists(data_path):
    print(f"错误: 数据文件不存在: {data_path}")
    print("请先运行 generate_multi_data.py 生成数据文件")
    exit(1)

# 加载数据
data = pd.read_csv(data_path)
print(f"原始数据形状: {data.shape}")
print(f"列名: {list(data.columns)}")

# 检查数据基本信息
print(f"\n数据类型:")
print(data.dtypes)

print(f"\n缺失值统计:")
print(data.isnull().sum())

print(f"\n前5行数据:")
print(data.head())

# 分离特征和目标
X = data.iloc[:, :-1].values  # 所有列除了最后一列
y = data.iloc[:, -1].values   # 最后一列作为目标

print(f"\n特征形状: X={X.shape}")
print(f"目标形状: y={y.shape}")

# 检查目标变量
n_classes = len(np.unique(y))
print(f"类别数: {n_classes}")
print(f"类别分布: {np.bincount(y)}")

# 2. 特征工程
print("\n2. 特征工程...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X_scaled)
print(f"原始特征数: {X.shape[1]}")
print(f"多项式特征数: {X_poly.shape[1]}")

# 3. 数据划分
print("\n3. 数据划分...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X_poly, y, test_size=0.1, random_state=42, shuffle=True, stratify=y
)
X_train, X_eval, y_train, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42, shuffle=True, stratify=y_temp
)  # 0.1111*0.9≈0.1
print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征 (80%)")
print(f"验证集: {X_eval.shape[0]} 样本, {X_eval.shape[1]} 特征 (10%)")
print(f"测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征 (10%)")
print(f"类别分布 - 训练集: {np.bincount(y_train)}")
print(f"类别分布 - 验证集: {np.bincount(y_eval)}")
print(f"类别分布 - 测试集: {np.bincount(y_test)}")

# 4. 模型训练
print("\n4. 模型训练...")
model = XGBoostNPU(
    max_depth=6,
    min_samples_split=10,
    learning_rate=0.1,
    n_estimators=100,
    lambda_param=0.1,
    task_type='multiclass',
    num_class=n_classes
)
print("开始训练...")
train_start_time = time.time()
model.train(X_train, y_train)
train_end_time = time.time()
train_time = train_end_time - train_start_time
print(f"训练完成! 训练时间: {train_time:.2f} 秒")

# 保存模型
print("\n保存模型...")
model_save_path = os.path.join(OUTPUT_DIR, "multi_class_model.pkl")
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ 模型已保存到: {model_save_path}")
except Exception as e:
    print(f"⚠ 模型保存失败: {e}")

# 5. 预测和评估
print("\n5. 预测和评估...")
def predict_and_time(model, X):
    start = time.time()
    preds = model.predict(X)
    end = time.time()
    return preds, end - start

print("训练集预测...")
train_preds_proba, train_pred_time = predict_and_time(model, X_train)
train_preds = np.argmax(train_preds_proba, axis=1)
print(f"训练集预测完成! 预测时间: {train_pred_time:.4f} 秒")

print("验证集预测...")
eval_preds_proba, eval_pred_time = predict_and_time(model, X_eval)
eval_preds = np.argmax(eval_preds_proba, axis=1)
print(f"验证集预测完成! 预测时间: {eval_pred_time:.4f} 秒")

print("测试集预测...")
test_preds_proba, test_pred_time = predict_and_time(model, X_test)
test_preds = np.argmax(test_preds_proba, axis=1)
print(f"测试集预测完成! 预测时间: {test_pred_time:.4f} 秒")

# 6. 评估指标
print("\n6. 评估指标...")
def evaluate_multiclass(y_true, y_pred, y_proba, dataset_name, pred_time=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(pd.get_dummies(y_true), y_proba, average='macro', multi_class='ovr')
    except Exception:
        auc = np.nan
    print(f"{dataset_name}结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
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
train_results = evaluate_multiclass(y_train, train_preds, train_preds_proba, "训练集", train_pred_time)
eval_results = evaluate_multiclass(y_eval, eval_preds, eval_preds_proba, "验证集", eval_pred_time)
test_results = evaluate_multiclass(y_test, test_preds, test_preds_proba, "测试集", test_pred_time)

# 7. 混淆矩阵分析
print("\n7. 混淆矩阵分析...")
def plot_confusion_matrix(y_true, y_pred, title, filename, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'类别{i}' for i in labels], 
                yticklabels=[f'类别{i}' for i in labels])
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  混淆矩阵已保存到: {filename}")

plot_confusion_matrix(y_test, test_preds, "测试集混淆矩阵", os.path.join(OUTPUT_DIR, "multi_test_confusion_matrix.png"), labels=list(range(n_classes)))
plot_confusion_matrix(y_train, train_preds, "训练集混淆矩阵", os.path.join(OUTPUT_DIR, "multi_train_confusion_matrix.png"), labels=list(range(n_classes)))

# 8. 保存结果
print("\n8. 保存结果...")
train_df = pd.DataFrame(X_train)
train_df['label'] = y_train
train_file_path = os.path.join(OUTPUT_DIR, 'multi_train.csv')
train_df.to_csv(train_file_path, index=False)
print(f"✓ 训练文件已保存: {train_file_path}")

eval_df = pd.DataFrame(X_eval)
eval_df['label'] = y_eval
eval_file_path = os.path.join(OUTPUT_DIR, 'multi_eval.csv')
eval_df.to_csv(eval_file_path, index=False)
print(f"✓ 验证文件已保存: {eval_file_path}")

test_df = pd.DataFrame(X_test)
test_df['label'] = y_test
test_df['pred_label'] = test_preds
test_file_path = os.path.join(OUTPUT_DIR, 'multi_test.csv')
test_df.to_csv(test_file_path, index=False)
print(f"✓ 测试文件已保存: {test_file_path} (含预测列)")

results_df = pd.DataFrame({
    '真实标签': y_test,
    '预测标签': test_preds,
    '预测概率_0': test_preds_proba[:, 0],
    '预测概率_1': test_preds_proba[:, 1],
    '预测概率_2': test_preds_proba[:, 2],
    '预测概率_3': test_preds_proba[:, 3],
    '预测正确': (y_test == test_preds)
})
results_file_path = os.path.join(OUTPUT_DIR, 'multi_classification_results.csv')
results_df.to_csv(results_file_path, index=False)
print(f"✓ 详细结果已保存到 {results_file_path}")

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
time_file_path = os.path.join(OUTPUT_DIR, 'multi_time_performance.csv')
pd.DataFrame([time_performance]).to_csv(time_file_path, index=False)
print(f"✓ 时间性能信息已保存到 {time_file_path}")

# 9. 可视化
print("\n9. 生成可视化图表...")
plt.figure(figsize=(15, 10))
# 子图1: 混淆矩阵
plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, test_preds, labels=list(range(n_classes)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'类别{i}' for i in range(n_classes)], 
            yticklabels=[f'类别{i}' for i in range(n_classes)])
plt.title('测试集混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
# 子图2: 各类别样本分布
plt.subplot(2, 3, 2)
plt.bar(range(n_classes), np.bincount(y_test), tick_label=[f'类别{i}' for i in range(n_classes)])
plt.title('测试集类别分布')
plt.xlabel('类别')
plt.ylabel('样本数')
# 子图3: 各类别预测分布
plt.subplot(2, 3, 3)
plt.bar(range(n_classes), np.bincount(test_preds), tick_label=[f'类别{i}' for i in range(n_classes)])
plt.title('测试集预测类别分布')
plt.xlabel('类别')
plt.ylabel('预测数')
# 子图4: 性能指标对比
plt.subplot(2, 3, 4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
train_values = [train_results['accuracy'], train_results['precision'], train_results['recall'], train_results['f1']]
test_values = [test_results['accuracy'], test_results['precision'], test_results['recall'], test_results['f1']]
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, train_values, width, label='训练集', alpha=0.8)
plt.bar(x + width/2, test_values, width, label='测试集', alpha=0.8)
plt.xlabel('指标')
plt.ylabel('分数')
plt.title('性能指标对比')
plt.xticks(x, metrics)
plt.legend()
# 子图5: 置信度分布
plt.subplot(2, 3, 5)
confidences = np.max(test_preds_proba, axis=1)
plt.hist(confidences, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('预测置信度')
plt.ylabel('样本数')
plt.title('测试集预测置信度分布')
plt.tight_layout()
results_image_path = os.path.join(OUTPUT_DIR, 'multi_classification_results.png')
plt.savefig(results_image_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 可视化图表已保存到 {results_image_path}")

# 10. 分类报告
print("\n10. 详细分类报告...")
print("测试集分类报告:")
print(classification_report(y_test, test_preds, target_names=[f'类别{i}' for i in range(n_classes)]))

# 11. 总结
print("\n" + "="*60)
print("多分类任务总结")
print("="*60)
print("主要特点:")
print("1. 多分类任务 - 4类，10万样本，15特征")
print("2. 从外部CSV文件读取数据")
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
print(f"训练集 - 准确率: {train_results['accuracy']:.4f}")
print(f"验证集 - 准确率: {eval_results['accuracy']:.4f}")
print(f"测试集 - 准确率: {test_results['accuracy']:.4f}")

print(f"\n输出文件:")
print(f"  {train_file_path} - 训练数据 (80%)")
print(f"  {eval_file_path} - 验证数据 (10%)")
print(f"  {test_file_path} - 测试数据 (10%) + 预测列")
print(f"  {model_save_path} - 训练好的模型")
print(f"  {time_file_path} - 时间性能信息")
print(f"  {results_image_path} - 可视化图表")
print(f"  {results_file_path} - 详细结果")

if test_results['accuracy'] > 0.8:
    print("\n✓ 性能优秀!")
elif test_results['accuracy'] > 0.6:
    print("\n✓ 性能良好!")
else:
    print("\n⚠ 性能需要进一步改进")

print("\n多分类任务完成!")
