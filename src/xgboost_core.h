// xgboost_core.h
#ifndef XGBOOST_CORE_H
#define XGBOOST_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

// 基本数据结构
typedef struct {
    int max_depth;
    int min_samples_split;
    float learning_rate;
    int n_estimators;
} XGBoostConfig;

typedef struct TreeNode {
    int feature_index;
    float split_value;
    float prediction;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

typedef struct {
    XGBoostConfig config;
    TreeNode** trees;
    int n_trees;
    aclrtContext context;
} XGBoostModel;

// 核心函数声明
XGBoostModel* create_xgboost_model(XGBoostConfig config);
int train_xgboost_model(XGBoostModel* model, float* X, float* y, int n_samples, int n_features);
float* predict_xgboost_model(XGBoostModel* model, float* X, int n_samples);
void free_xgboost_model(XGBoostModel* model);

// 内部函数声明
float predict_tree(TreeNode* tree, float* X, int n_features);
int split_tree_node(TreeNode* node, float* X, float* y, float* weights, int n_samples, int n_features, 
                   int depth, XGBoostConfig* config);
TreeNode* create_tree_node(int feature_index, float split_value, float prediction);

#ifdef __cplusplus
}
#endif

#endif // XGBOOST_CORE_H