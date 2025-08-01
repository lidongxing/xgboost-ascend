// xgboost_core.h
#ifndef XGBOOST_CORE_H
#define XGBOOST_CORE_H

#include <acl/acl.h>

#ifdef __cplusplus
extern "C" {
#endif

// 任务类型定义
#define TASK_BINARY 0
#define TASK_MULTICLASS 1
#define TASK_REGRESSION 2

// 树节点结构
typedef struct TreeNode {
    int feature_index;
    float split_value;
    float weight;        // 替换原来的 prediction
    int is_leaf;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

// XGBoost配置结构
typedef struct XGBoostConfig {
    int max_depth;
    int min_samples_split;
    float learning_rate;
    int n_estimators;
    float lambda_param;  // 注意这里是 lambda_param 而不是 lambda
    int task_type;
    int num_class;
} XGBoostConfig;

// XGBoost模型结构
typedef struct {
    int max_depth;
    int min_samples_split;
    float learning_rate;
    int n_estimators;
    float lambda;
    int task_type;
    int num_class;
    TreeNode** trees;    // 对于多分类，每个类别都有一组树
    int n_trees;         // 当前树的数量
    XGBoostConfig config;// 保存配置
    aclrtContext context;// NPU上下文
} XGBoostModel;

// 函数声明
XGBoostModel* create_xgboost_model(XGBoostConfig config);
int train(XGBoostModel* model, float* X, float* y, int n_samples, int n_features);
float* predict(XGBoostModel* model, float* X, int n_samples, int n_features);
void free_xgboost_model(XGBoostModel* model);

// 内部辅助函数声明
float calculate_gradient(float y_true, float y_pred, int task_type);
float calculate_hessian(float y_true, float y_pred, int task_type);
float calculate_leaf_weight(float sum_grad, float sum_hess, float lambda, int task_type);
float calculate_split_gain(float sum_grad, float sum_hess, float left_grad, float left_hess,
                         float right_grad, float right_hess, float lambda, int task_type);
void predict_tree(TreeNode* node, float* features, float* pred, int task_type);
void predict_multiclass(XGBoostModel* model, float* features, float* pred, int num_class);

// NPU相关函数声明
int init_acl(XGBoostModel* model);
void finalize_acl(XGBoostModel* model);

#ifdef __cplusplus
}
#endif

#endif // XGBOOST_CORE_H
