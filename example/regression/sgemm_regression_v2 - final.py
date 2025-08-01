import numpy as np
import pandas as pd
from xgboost_npu import XGBoostNPU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
print("SGemm性能预测 - 基于改进版V2")
print("=" * 60)

# 1. 数据加载和基础分析
print("1. 数据加载和基础分析...")
data = pd.read_csv('sgemm_product.csv')

# 分离特征和标签
# 特征列：前14列 (MWG, NWG, KWG, MDIMC, NDIMC, MDIMA, NDIMB, KWI, VWM, VWN, STRM, STRN, SA, SB)
# 目标列：Run1 (ms)
X = data.iloc[:, :14].values  # 前14列作为特征
y = data.iloc[:, 14].values   # Run1 (ms) 作为目标

print(f"原始数据形状: X={X.shape}, y={y.shape}")
print(f"特征列: {list(data.columns[:14])}")
print(f"目标列: {data.columns[14]}")

print(f"\n目标变量统计:")
print(f"  范围: [{np.min(y):.2f}, {np.max(y):.2f}]")
print(f"  均值: {np.mean(y):.2f}")
print(f"  中位数: {np.median(y):.2f}")
print(f"  标准差: {np.std(y):.2f}")
print(f"  偏度: {pd.Series(y).skew():.2f}")

# 2. 目标变量变换
print("\n2. 目标变量变换...")
# 使用Box-Cox变换处理右偏分布
from scipy.stats import boxcox
y_transformed, lambda_param = boxcox(y + 1)  # +1避免负值

print(f"Box-Cox变换参数 λ: {lambda_param:.4f}")
print(f"变换后统计:")
print(f"  范围: [{np.min(y_transformed):.2f}, {np.max(y_transformed):.2f}]")
print(f"  均值: {np.mean(y_transformed):.2f}")
print(f"  标准差: {np.std(y_transformed):.2f}")
print(f"  偏度: {pd.Series(y_transformed).skew():.2f}")

# 3. 特征工程
print("\n3. 特征工程...")
# 使用PowerTransformer处理特征偏度
from sklearn.preprocessing import PowerTransformer
feature_transformer = PowerTransformer(method='yeo-johnson')
X_transformed = feature_transformer.fit_transform(X)

# 添加多项式特征（二次项）
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X_transformed)

print(f"原始特征数: {X.shape[1]}")
print(f"变换后特征数: {X_transformed.shape[1]}")
print(f"多项式特征数: {X_poly.shape[1]}")

# 4. 数据划分
print("\n4. 数据划分...")
# 重要：使用shuffle=True确保数据正确打乱
# 首先划分训练集和临时集 (80% vs 20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_poly, y_transformed, test_size=0.1, random_state=42, shuffle=True
)

# 然后从临时集中划分训练集和验证集 (80% vs 20% of remaining 90% = 72% vs 18%)
X_train, X_eval, y_train, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, shuffle=True  # 0.125 * 0.9 = 0.1125 ≈ 0.1
)

print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征 (80%)")
print(f"验证集: {X_eval.shape[0]} 样本, {X_eval.shape[1]} 特征 (10%)")
print(f"测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征 (10%)")
print("✓ 数据已正确打乱 (shuffle=True)")
print("✓ 划分比例: 训练集80% : 验证集10% : 测试集10%")

# 5. 模型训练
print("\n5. 模型训练...")
model = XGBoostNPU(
    max_depth=6,           # 增加深度
    min_samples_split=5,   # 减少分裂阈值
    learning_rate=0.03,    # 进一步降低学习率
    n_estimators=200,      # 增加树的数量
    lambda_param=0.01,     # 减少正则化
    task_type='regression'
)

print("开始训练...")
train_start_time = time.time()
model.train(X_train, y_train)
train_end_time = time.time()
train_time = train_end_time - train_start_time
print(f"训练完成! 训练时间: {train_time:.2f} 秒")

# 保存模型
print("\n保存模型...")
model_save_path = "sgemm_regression_model.pkl"
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ 模型已保存到: {model_save_path}")
except Exception as e:
    print(f"⚠ 模型保存失败: {e}")

# 定义逆变换函数
def inverse_boxcox(y_transformed, lambda_param):
    if lambda_param == 0:
        return np.exp(y_transformed) - 1
    else:
        return (y_transformed * lambda_param + 1) ** (1 / lambda_param) - 1

# 验证集评估
print("\n验证集评估...")
eval_start_time = time.time()
eval_preds_transformed = model.predict(X_eval)
eval_end_time = time.time()
eval_pred_time = eval_end_time - eval_start_time
print(f"验证集预测完成! 预测时间: {eval_pred_time:.4f} 秒")

eval_preds = inverse_boxcox(eval_preds_transformed, lambda_param)  # 转换回原始尺度
eval_y_original = inverse_boxcox(y_eval, lambda_param)

eval_r2 = r2_score(eval_y_original, eval_preds)
eval_rmse = np.sqrt(mean_squared_error(eval_y_original, eval_preds))
eval_mae = mean_absolute_error(eval_y_original, eval_preds)
eval_mape = np.mean(np.abs((eval_y_original - eval_preds) / eval_y_original)) * 100

print(f"验证集 R²: {eval_r2:.4f}")
print(f"验证集 RMSE: {eval_rmse:.2f}")
print(f"验证集 MAE: {eval_mae:.2f}")
print(f"验证集 MAPE: {eval_mape:.2f}%")

# 6. 预测和评估
print("\n6. 预测和评估...")

# 训练集预测
print("训练集预测...")
train_pred_start_time = time.time()
train_preds_transformed = model.predict(X_train)
train_pred_end_time = time.time()
train_pred_time = train_pred_end_time - train_pred_start_time
print(f"训练集预测完成! 预测时间: {train_pred_time:.4f} 秒")

train_preds = inverse_boxcox(train_preds_transformed, lambda_param)  # 转换回原始尺度
train_y_original = inverse_boxcox(y_train, lambda_param)

# 测试集预测
print("测试集预测...")
test_pred_start_time = time.time()
test_preds_transformed = model.predict(X_test)
test_pred_end_time = time.time()
test_pred_time = test_pred_end_time - test_pred_start_time
print(f"测试集预测完成! 预测时间: {test_pred_time:.4f} 秒")

test_preds = inverse_boxcox(test_preds_transformed, lambda_param)  # 转换回原始尺度
test_y_original = inverse_boxcox(y_test, lambda_param)

# 7. 评估指标
print("\n7. 评估指标...")
print("训练集结果:")
train_r2 = r2_score(train_y_original, train_preds)
train_rmse = np.sqrt(mean_squared_error(train_y_original, train_preds))
train_mae = mean_absolute_error(train_y_original, train_preds)
train_mape = np.mean(np.abs((train_y_original - train_preds) / train_y_original)) * 100

print(f"  R²: {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.2f}")
print(f"  MAE: {train_mae:.2f}")
print(f"  MAPE: {train_mape:.2f}%")

print("\n验证集结果:")
print(f"  R²: {eval_r2:.4f}")
print(f"  RMSE: {eval_rmse:.2f}")
print(f"  MAE: {eval_mae:.2f}")
print(f"  MAPE: {eval_mape:.2f}%")

print("\n测试集结果:")
test_r2 = r2_score(test_y_original, test_preds)
test_rmse = np.sqrt(mean_squared_error(test_y_original, test_preds))
test_mae = mean_absolute_error(test_y_original, test_preds)
test_mape = np.mean(np.abs((test_y_original - test_preds) / test_y_original)) * 100

print(f"  R²: {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  MAE: {test_mae:.2f}")
print(f"  MAPE: {test_mape:.2f}%")

# 8. 预测值分析
print("\n8. 预测值分析...")
print(f"原始目标变量范围: [{np.min(test_y_original):.2f}, {np.max(test_y_original):.2f}]")
print(f"预测值范围: [{np.min(test_preds):.2f}, {np.max(test_preds):.2f}]")
print(f"预测值均值: {np.mean(test_preds):.2f}")
print(f"预测值标准差: {np.std(test_preds):.2f}")

# 9. 相对误差分析
print("\n9. 相对误差分析...")
relative_errors = np.abs((test_y_original - test_preds) / test_y_original) * 100
print(f"相对误差统计:")
print(f"  均值: {np.mean(relative_errors):.2f}%")
print(f"  中位数: {np.median(relative_errors):.2f}%")
print(f"  90%分位数: {np.percentile(relative_errors, 90):.2f}%")
print(f"  95%分位数: {np.percentile(relative_errors, 95):.2f}%")

# 10. 保存结果
print("\n10. 保存结果...")

# 准备原始数据（用于保存到文件）
original_data = pd.read_csv('sgemm_product.csv')

# 获取原始数据的索引
all_indices = np.arange(len(original_data))
_, test_indices = train_test_split(all_indices, test_size=0.1, random_state=42, shuffle=True)
temp_indices = np.setdiff1d(all_indices, test_indices)
train_indices, eval_indices = train_test_split(temp_indices, test_size=0.125, random_state=42, shuffle=True)

# 创建训练文件
train_data = original_data.iloc[train_indices].copy()
train_data.to_csv('sgemm_train.csv', index=False)
print("✓ 训练文件已保存: sgemm_train.csv")

# 创建验证文件
eval_data = original_data.iloc[eval_indices].copy()
eval_data.to_csv('sgemm_eval.csv', index=False)
print("✓ 验证文件已保存: sgemm_eval.csv")

# 创建测试文件（包含预测列）
test_data = original_data.iloc[test_indices].copy()
test_data['Run1_pred'] = test_preds  # 添加预测列
test_data.to_csv('sgemm_test.csv', index=False)
print("✓ 测试文件已保存: sgemm_test.csv (包含预测列)")

# 保存详细结果
results_df = pd.DataFrame({
    '真实值': test_y_original,
    '预测值': test_preds,
    '绝对误差': np.abs(test_preds - test_y_original),
    '相对误差(%)': relative_errors
})

results_df.to_csv('sgemm_regression_results.csv', index=False)
print("✓ 详细结果已保存到 sgemm_regression_results.csv")

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
time_df.to_csv('sgemm_time_performance.csv', index=False)
print("✓ 时间性能信息已保存到 sgemm_time_performance.csv")

# 显示文件信息
print(f"\n文件信息:")
print(f"  sgemm_train.csv: {len(train_data)} 样本 (训练用)")
print(f"  sgemm_eval.csv: {len(eval_data)} 样本 (验证用)")
print(f"  sgemm_test.csv: {len(test_data)} 样本 (测试用，包含预测列)")
print(f"  sgemm_regression_model.pkl: 训练好的模型文件")
print(f"  sgemm_time_performance.csv: 时间性能信息")

# 11. 可视化
print("\n11. 生成可视化图表...")
plt.figure(figsize=(15, 10))

# 子图1: 预测值 vs 真实值
plt.subplot(2, 3, 1)
plt.scatter(test_y_original, test_preds, alpha=0.6)
plt.plot([test_y_original.min(), test_y_original.max()], 
         [test_y_original.min(), test_y_original.max()], 'r--', lw=2)
plt.xlabel('真实值 (Run1 ms)')
plt.ylabel('预测值 (Run1_pred ms)')
plt.title(f'预测值 vs 真实值\nR² = {test_r2:.4f}')

# 子图2: 残差图
plt.subplot(2, 3, 2)
residuals = test_preds - test_y_original
plt.scatter(test_preds, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')

# 子图3: 相对误差分布
plt.subplot(2, 3, 3)
plt.hist(relative_errors, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('相对误差 (%)')
plt.ylabel('频数')
plt.title('相对误差分布')

# 子图4: 目标变量分布对比
plt.subplot(2, 3, 4)
plt.hist(test_y_original, bins=30, alpha=0.7, label='真实值', edgecolor='black')
plt.hist(test_preds, bins=30, alpha=0.7, label='预测值', edgecolor='black')
plt.xlabel('Run1 (ms)')
plt.ylabel('频数')
plt.title('目标变量分布对比')
plt.legend()

# 子图5: 误差分布
plt.subplot(2, 3, 5)
plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('绝对误差')
plt.ylabel('频数')
plt.title('绝对误差分布')

# 子图6: 性能指标对比
plt.subplot(2, 3, 6)
metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
train_values = [train_r2, train_rmse, train_mae, train_mape]
test_values = [test_r2, test_rmse, test_mae, test_mape]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_values, width, label='训练集', alpha=0.8)
plt.bar(x + width/2, test_values, width, label='测试集', alpha=0.8)
plt.xlabel('指标')
plt.ylabel('值')
plt.title('性能指标对比')
plt.xticks(x, metrics)
plt.legend()

plt.tight_layout()
plt.savefig('sgemm_regression_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 可视化图表已保存到 sgemm_regression_results.png")

# 12. 总结
print("\n" + "="*60)
print("SGemm回归预测总结")
print("="*60)
print("主要改进:")
print("1. Box-Cox变换 - 更好的分布变换")
print("2. PowerTransformer - 特征偏度处理")
print("3. 多项式特征 - 增加特征交互")
print("4. 更深的模型 - 增加学习能力")
print("5. 更多树数量 - 提高模型复杂度")
print("6. 三文件划分 - 训练/验证/测试 (8:1:1)")
print("7. 模型保存 - 可重复使用训练好的模型")
print("8. 时间记录 - 记录训练和预测时间")

print(f"\n时间性能:")
print(f"训练时间: {train_time:.2f} 秒")
print(f"训练集预测时间: {train_pred_time:.4f} 秒 (样本数: {len(X_train)})")
print(f"验证集预测时间: {eval_pred_time:.4f} 秒 (样本数: {len(X_eval)})")
print(f"测试集预测时间: {test_pred_time:.4f} 秒 (样本数: {len(X_test)})")
print(f"总预测时间: {train_pred_time + eval_pred_time + test_pred_time:.4f} 秒 (总样本数: {len(X_train) + len(X_eval) + len(X_test)})")

print(f"\n最终性能:")
print(f"训练集 R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}")
print(f"验证集 R²: {eval_r2:.4f}, RMSE: {eval_rmse:.2f}")
print(f"测试集 R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")

print(f"\n输出文件:")
print(f"  sgemm_train.csv - 训练数据 (80%)")
print(f"  sgemm_eval.csv - 验证数据 (10%)")
print(f"  sgemm_test.csv - 测试数据 (10%) + Run1_pred列")
print(f"  sgemm_regression_model.pkl - 训练好的模型")
print(f"  sgemm_time_performance.csv - 时间性能信息")
print(f"  sgemm_regression_results.png - 可视化图表")
print(f"  sgemm_regression_results.csv - 详细结果")

if test_r2 > 0.8:
    print("\n✓ 性能优秀!")
elif test_r2 > 0.6:
    print("\n✓ 性能良好!")
elif test_r2 > 0.4:
    print("\n✓ 性能可接受!")
else:
    print("\n⚠ 性能需要进一步改进")

print("\nSGemm回归预测完成!") 