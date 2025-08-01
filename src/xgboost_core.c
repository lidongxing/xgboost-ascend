#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include <acl/acl.h>  // Ascend NPU API
#include "xgboost_core.h"
#include <math.h>  // 新增：添加 math.h 头文件，用于使用 fmaxf、fminf、logf、expf 等数学函数
#include <float.h>

// 添加浮点数精度检查宏定义
#define CHECK_FLOAT(x) (isnan(x) || isinf(x) ? 0.0f : x)
#define SIGMOID(x) (1.0f / (1.0f + expf(-CHECK_FLOAT(x))))

// 定义任务类型
#define TASK_BINARY 0
#define TASK_MULTICLASS 1
#define TASK_REGRESSION 2

// 计算不同任务的梯度
float calculate_gradient(float y_true, float y_pred, int task_type) {
    switch (task_type) {
        case TASK_BINARY:
            // 二分类：交叉熵损失
            // y_pred 是原始分数，需要转换为概率
            float prob = 1.0f / (1.0f + expf(-y_pred));
            return prob - y_true;
        
        case TASK_MULTICLASS:
            // 多分类：softmax交叉熵损失
            return y_pred - y_true;
        
        case TASK_REGRESSION:
            // 回归：均方误差损失
            // 添加数值稳定性检查
            float diff = y_pred - y_true;
            if (isnan(diff) || isinf(diff)) {
                return 0.0f;
            }
            return 2.0f * diff;  // MSE损失的梯度
        
        default:
            return 0.0f;
    }
}

// 计算不同任务的二阶导数
float calculate_hessian(float y_true, float y_pred, int task_type) {
    switch (task_type) {
        case TASK_BINARY:
            // 二分类：使用sigmoid概率
            float prob = 1.0f / (1.0f + expf(-y_pred));
            return prob * (1.0f - prob);
        
        case TASK_MULTICLASS:
            return y_pred * (1.0f - y_pred);
        
        case TASK_REGRESSION:
            // 回归：MSE损失的二阶导数
            // 添加数值稳定性检查
            float hess = 2.0f;
            if (isnan(hess) || isinf(hess)) {
                return 1.0f;  // 返回一个安全的默认值
            }
            return hess;
        
        default:
            return 1.0f;
    }
}

// 计算叶子节点的预测值
float calculate_leaf_weight(float sum_grad, float sum_hess, float lambda, int task_type) {
    // 添加数值稳定性检查
    if (isnan(sum_grad) || isinf(sum_grad) || isnan(sum_hess) || isinf(sum_hess)) {
        return 0.0f;
    }
    
    switch (task_type) {
        case TASK_BINARY:
            // 添加正则化和数值稳定性
            if (fabs(sum_hess) < 1e-6f) return 0.0f;
            float weight = -sum_grad / (sum_hess + lambda);
            // 限制权重范围，防止过大的值
            if (isnan(weight) || isinf(weight)) return 0.0f;
            if (weight > 100.0f) weight = 100.0f;
            if (weight < -100.0f) weight = -100.0f;
            return weight;
        
        case TASK_REGRESSION:
            if (fabs(sum_hess) < 1e-6f) return 0.0f;
            weight = -sum_grad / (sum_hess + lambda);
            // 限制权重范围，防止过大的值
            if (isnan(weight) || isinf(weight)) return 0.0f;
            if (weight > 100.0f) weight = 100.0f;
            if (weight < -100.0f) weight = -100.0f;
            return weight;
        
        default:
            return 0.0f;
    }
}

// 计算分裂增益
float calculate_split_gain(float sum_grad, float sum_hess, 
                         float left_grad, float left_hess,
                         float right_grad, float right_hess,
                         float lambda, int task_type) {
    // 添加数值稳定性检查
    if (isnan(sum_grad) || isinf(sum_grad) || isnan(sum_hess) || isinf(sum_hess) ||
        isnan(left_grad) || isinf(left_grad) || isnan(left_hess) || isinf(left_hess) ||
        isnan(right_grad) || isinf(right_grad) || isnan(right_hess) || isinf(right_hess)) {
        return -FLT_MAX;  // 返回一个很小的值，表示这个分裂不好
    }
    
    // 父节点的得分
    float parent_score = 0.0f;
    if (fabs(sum_hess + lambda) > 1e-6f) {
        parent_score = (sum_grad * sum_grad) / (2 * (sum_hess + lambda));
        if (isnan(parent_score) || isinf(parent_score)) {
            parent_score = 0.0f;
        }
    }
    
    // 左子节点的得分
    float left_score = 0.0f;
    if (fabs(left_hess + lambda) > 1e-6f) {
        left_score = (left_grad * left_grad) / (2 * (left_hess + lambda));
        if (isnan(left_score) || isinf(left_score)) {
            left_score = 0.0f;
        }
    }
    
    // 右子节点的得分
    float right_score = 0.0f;
    if (fabs(right_hess + lambda) > 1e-6f) {
        right_score = (right_grad * right_grad) / (2 * (right_hess + lambda));
        if (isnan(right_score) || isinf(right_score)) {
            right_score = 0.0f;
        }
    }
    
    // 增益 = 左子节点得分 + 右子节点得分 - 父节点得分
    float gain = left_score + right_score - parent_score;
    
    // 最终检查
    if (isnan(gain) || isinf(gain)) {
        return -FLT_MAX;
    }
    
    return gain;
}

// 修改预测树节点的函数
void predict_tree(TreeNode* node, float* features, float* pred, int task_type) {
    if (!node) return;
    
    if (node->is_leaf) {
        // 直接返回节点权重，不做sigmoid变换
        *pred = node->weight;
        return;
    }
    
    if (features[node->feature_index] <= node->split_value) {
        predict_tree(node->left, features, pred, task_type);
    } else {
        predict_tree(node->right, features, pred, task_type);
    }
}

// 多分类预测
void predict_multiclass(XGBoostModel* model, float* features, float* pred, int num_class) {
    if (!model || !features || !pred || num_class <= 0) {
        printf("Error: Invalid parameters in predict_multiclass\n");
        return;
    }
    
    // 计算每个类别的原始分数
    for (int c = 0; c < num_class; c++) {
        pred[c] = 0.0f;
        int trees_per_class = model->n_trees / num_class;
        
        // 遍历该类别的所有树
        for (int i = 0; i < trees_per_class; i++) {
            int tree_index = i * num_class + c;
            if (tree_index < model->n_trees && model->trees && model->trees[tree_index]) {
            float tree_pred = 0.0f;
                predict_tree(model->trees[tree_index], features, &tree_pred, TASK_REGRESSION);
                pred[c] += tree_pred;
        }
        }
    }
    
    // 使用温度缩放的softmax来平衡概率分布
    float max_score = -FLT_MAX;
    for (int c = 0; c < num_class; c++) {
        if (pred[c] > max_score) {
            max_score = pred[c];
    }
    }
    
    // 使用温度参数来软化softmax
    float temperature = 2.0f; // 降低温度使分布更敏感
    
    float sum_exp = 0.0f;
    for (int c = 0; c < num_class; c++) {
        // 应用温度缩放的softmax：exp((score - max_score) / temperature)
        pred[c] = expf((pred[c] - max_score) / temperature);
        sum_exp += pred[c];
    }
    
    // 归一化概率
    if (sum_exp > 0.0f) {
        for (int c = 0; c < num_class; c++) {
            pred[c] /= sum_exp;
        }
    } else {
        // 如果sum为0，平均分配概率
        for (int c = 0; c < num_class; c++) {
            pred[c] = 1.0f / num_class;
        }
    }
}

// Initialize ACL and create context
int init_acl(XGBoostModel* model) {
    printf("Initializing ACL...\n");
    aclError ret = aclInit(NULL);
    if (ret != ACL_SUCCESS) {
        printf("Failed to initialize ACL, error code: %d\n", ret);
        return -1;
    }
    printf("ACL initialized successfully\n");

    ret = aclrtSetDevice(0);  // Use device 0
    if (ret != ACL_SUCCESS) {
        printf("Failed to set device, error code: %d\n", ret);
        aclFinalize();  // 确保在设置设备失败时释放 ACL 资源
        return -1;
    }
    printf("Device set successfully\n");

    ret = aclrtCreateContext(&model->context, 0);
    if (ret != ACL_SUCCESS) {
        printf("Failed to create context, error code: %d\n", ret);
        aclrtResetDevice(0);  // 新增：重置设备
        aclFinalize();  // 新增：在创建上下文失败时释放 ACL 资源
        return -1;
    }
    printf("Context created successfully\n");

    return 0;
}

// Create a tree node
TreeNode* create_tree_node(int feature_index, float split_value, float weight, int is_leaf) {
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    if (!node) {
        printf("Failed to allocate memory for tree node\n");
        return NULL;
    }

    node->feature_index = feature_index;
    node->split_value = split_value;
    node->weight = weight;
    node->is_leaf = is_leaf;
    node->left = NULL;
    node->right = NULL;

    return node;
}

// 添加快速排序函数用于特征值排序
void quick_sort(float* arr, int* indices, int left, int right) {
    if (left < right) {
        float pivot = arr[(left + right) / 2];
        int i = left - 1;
        int j = right + 1;
        
        while (1) {
            do { i++; } while (arr[i] < pivot);
            do { j--; } while (arr[j] > pivot);
            
            if (i >= j) break;
            
            // 交换值
            float temp_val = arr[i];
            arr[i] = arr[j];
            arr[j] = temp_val;
            
            // 交换索引
            int temp_idx = indices[i];
            indices[i] = indices[j];
            indices[j] = temp_idx;
        }
        
        quick_sort(arr, indices, left, j);
        quick_sort(arr, indices, j + 1, right);
    }
}

// 优化的树分裂函数
void split_tree_node(TreeNode* node, float* X, float* gradients, float* hessians,
                    int n_samples, int n_features, int depth,
                    XGBoostConfig* config) {
    if (depth >= config->max_depth || n_samples < config->min_samples_split) {
        node->is_leaf = 1;
        // 计算叶子节点的权重
        float sum_grad = 0.0f, sum_hess = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            sum_grad += gradients[i];
            sum_hess += hessians[i];
        }
        
        // 添加正则化和数值稳定性
        if (fabs(sum_hess + config->lambda_param) > 1e-6f) {
            node->weight = -sum_grad / (sum_hess + config->lambda_param);
        } else {
            node->weight = 0.0f;
        }
        
        printf("Leaf node created: depth=%d, samples=%d, weight=%f\n",
               depth, n_samples, node->weight);
        return;
    }

    float best_gain = -FLT_MAX;
    int best_feature = -1;
    float best_split = 0.0f;
    float best_left_weight = 0.0f;
    float best_right_weight = 0.0f;

    // 预分配内存，避免重复分配
    float* feature_values = (float*)malloc(n_samples * sizeof(float));
    int* sample_indices = (int*)malloc(n_samples * sizeof(int));
    float* sorted_gradients = (float*)malloc(n_samples * sizeof(float));
    float* sorted_hessians = (float*)malloc(n_samples * sizeof(float));
    
    if (!feature_values || !sample_indices || !sorted_gradients || !sorted_hessians) {
        printf("Error: Failed to allocate memory for tree splitting\n");
        if (feature_values) free(feature_values);
        if (sample_indices) free(sample_indices);
        if (sorted_gradients) free(sorted_gradients);
        if (sorted_hessians) free(sorted_hessians);
        return;
    }

    // 遍历所有特征寻找最佳分裂点
    for (int feature = 0; feature < n_features; feature++) {
        // 复制特征值和初始化索引
        for (int i = 0; i < n_samples; i++) {
            feature_values[i] = X[i * n_features + feature];
            sample_indices[i] = i;
        }
        
        // 使用快速排序排序特征值
        quick_sort(feature_values, sample_indices, 0, n_samples - 1);
        
        // 根据排序后的索引重新排列梯度和二阶导数
        for (int i = 0; i < n_samples; i++) {
            sorted_gradients[i] = gradients[sample_indices[i]];
            sorted_hessians[i] = hessians[sample_indices[i]];
        }
        
        // 计算累积梯度和二阶导数
        float* cum_grad = (float*)malloc((n_samples + 1) * sizeof(float));
        float* cum_hess = (float*)malloc((n_samples + 1) * sizeof(float));
        
        if (!cum_grad || !cum_hess) {
            free(cum_grad);
            free(cum_hess);
            continue;
        }
        
        cum_grad[0] = 0.0f;
        cum_hess[0] = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            cum_grad[i + 1] = cum_grad[i] + sorted_gradients[i];
            cum_hess[i + 1] = cum_hess[i] + sorted_hessians[i];
        }
        
        // 尝试不同的分裂点（使用采样来减少计算量）
        int step = n_samples > 10000 ? n_samples / 500 : 1; // 减少采样间隔，提高精度
        for (int i = 0; i < n_samples; i += step) {
            if (i == 0 || i == n_samples - 1) continue; // 跳过边界点
            
            float split_value = feature_values[i];
            
            // 使用累积和快速计算左右子节点的统计量
            float left_grad = cum_grad[i];
            float left_hess = cum_hess[i];
            float right_grad = cum_grad[n_samples] - cum_grad[i];
            float right_hess = cum_hess[n_samples] - cum_hess[i];
            
            int left_count = i;
            int right_count = n_samples - i;
            
            // 如果分裂是有效的
            if (left_count >= config->min_samples_split && right_count >= config->min_samples_split) {
                float gain = calculate_split_gain(
                    left_grad + right_grad, left_hess + right_hess,
                    left_grad, left_hess,
                    right_grad, right_hess,
                    config->lambda_param, config->task_type
                );
                
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = feature;
                    best_split = split_value;
                    best_left_weight = -left_grad / (left_hess + config->lambda_param);
                    best_right_weight = -right_grad / (right_hess + config->lambda_param);
                }
            }
        }
        
        free(cum_grad);
        free(cum_hess);
    }
    
    // 释放临时内存
        free(feature_values);
    free(sample_indices);
    free(sorted_gradients);
    free(sorted_hessians);

    // 如果找到了有效的分裂点
    if (best_feature != -1 && best_gain > 0.0f) {
        node->feature_index = best_feature;
        node->split_value = best_split;
        node->is_leaf = 0;
        
        printf("Split node created: depth=%d, feature=%d, split=%f, gain=%f\n",
               depth, best_feature, best_split, best_gain);
        
        // 创建子节点
        node->left = create_tree_node(-1, 0.0f, best_left_weight, 1);
        node->right = create_tree_node(-1, 0.0f, best_right_weight, 1);
        
        // 递归构建子树
        // 为左子树准备数据
        int left_count = 0;
        for (int i = 0; i < n_samples; i++) {
            if (X[i * n_features + best_feature] <= best_split) {
                left_count++;
            }
        }
        
        if (left_count > 0) {
            float* left_X = (float*)malloc(left_count * n_features * sizeof(float));
            float* left_gradients = (float*)malloc(left_count * sizeof(float));
            float* left_hessians = (float*)malloc(left_count * sizeof(float));
            
            if (left_X && left_gradients && left_hessians) {
                int left_idx = 0;
                for (int i = 0; i < n_samples; i++) {
                    if (X[i * n_features + best_feature] <= best_split) {
                        memcpy(&left_X[left_idx * n_features], &X[i * n_features], n_features * sizeof(float));
                        left_gradients[left_idx] = gradients[i];
                        left_hessians[left_idx] = hessians[i];
                        left_idx++;
                    }
                }
                
                split_tree_node(node->left, left_X, left_gradients, left_hessians,
                              left_count, n_features, depth + 1, config);
                
                free(left_X);
                free(left_gradients);
                free(left_hessians);
            }
        }
        
        // 为右子树准备数据
        int right_count = n_samples - left_count;
        if (right_count > 0) {
            float* right_X = (float*)malloc(right_count * n_features * sizeof(float));
            float* right_gradients = (float*)malloc(right_count * sizeof(float));
            float* right_hessians = (float*)malloc(right_count * sizeof(float));
            
            if (right_X && right_gradients && right_hessians) {
                int right_idx = 0;
                for (int i = 0; i < n_samples; i++) {
                    if (X[i * n_features + best_feature] > best_split) {
                        memcpy(&right_X[right_idx * n_features], &X[i * n_features], n_features * sizeof(float));
                        right_gradients[right_idx] = gradients[i];
                        right_hessians[right_idx] = hessians[i];
                        right_idx++;
                    }
                }
                
                split_tree_node(node->right, right_X, right_gradients, right_hessians,
                              right_count, n_features, depth + 1, config);
                
                free(right_X);
                free(right_gradients);
                free(right_hessians);
            }
        }
    } else {
        // 如果没有找到好的分裂点，设为叶子节点
        node->is_leaf = 1;
        float sum_grad = 0.0f, sum_hess = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            sum_grad += gradients[i];
            sum_hess += hessians[i];
        }
        node->weight = -sum_grad / (sum_hess + config->lambda_param);
    }
}

// Create XGBoost model
XGBoostModel* create_xgboost_model(XGBoostConfig config) {
    printf("Creating XGBoost model with config:\n");
    printf("- max_depth: %d\n", config.max_depth);
    printf("- min_samples_split: %d\n", config.min_samples_split);
    printf("- learning_rate: %f\n", config.learning_rate);
    printf("- n_estimators: %d\n", config.n_estimators);
    printf("- lambda_param: %f\n", config.lambda_param);
    printf("- task_type: %d\n", config.task_type);
    printf("- num_class: %d\n", config.num_class);

    // 验证配置参数
    if (config.max_depth <= 0 || config.min_samples_split <= 0 || 
        config.learning_rate <= 0 || config.n_estimators <= 0) {
        printf("Invalid configuration parameters\n");
        return NULL;
    }

    // 分配模型内存
    XGBoostModel* model = (XGBoostModel*)calloc(1, sizeof(XGBoostModel));
    if (!model) {
        printf("Failed to allocate memory for model\n");
        return NULL;
    }

    // 分配树数组内存
    int total_trees;
    if (config.task_type == TASK_MULTICLASS) {
        total_trees = config.n_estimators * config.num_class;
    } else {
        total_trees = config.n_estimators;
    }
    
    model->trees = (TreeNode**)calloc(total_trees, sizeof(TreeNode*));
    if (!model->trees) {
        printf("Failed to allocate memory for trees array\n");
        free(model);
        return NULL;
    }

    // 初始化所有成员
    model->config = config;
    model->max_depth = config.max_depth;
    model->min_samples_split = config.min_samples_split;
    model->learning_rate = config.learning_rate;
    model->n_estimators = config.n_estimators;
    model->lambda = config.lambda_param;
    model->task_type = config.task_type;
    model->num_class = config.num_class;
    model->n_trees = 0;
    model->context = NULL;

    // 初始化树指针数组
    for (int i = 0; i < total_trees; i++) {
        model->trees[i] = NULL;
    }

    // 初始化 NPU
    if (init_acl(model) != 0) {
        printf("Failed to initialize ACL\n");
        free(model->trees);
        free(model);
        return NULL;
    }

    printf("XGBoost model created successfully\n");
    return model;
}

// 修改预测函数
float* predict(XGBoostModel* model, float* X, int n_samples, int n_features) {
    // 根据任务类型分配内存
    int output_size;
    if (model->config.task_type == TASK_MULTICLASS) {
        output_size = n_samples * model->config.num_class;
    } else {
        output_size = n_samples;
    }
    
    float* predictions = (float*)malloc(output_size * sizeof(float));
    if (!predictions) {
        printf("Error: Failed to allocate memory for predictions\n");
        return NULL;
    }

    // 打印模型配置信息
    printf("Model configuration:\n");
    printf("Task type: %d\n", model->config.task_type);
    printf("Number of trees: %d\n", model->n_trees);
    printf("Learning rate: %f\n", model->config.learning_rate);

    if (model->config.task_type == TASK_MULTICLASS) {
        // 多分类预测
        // 初始化预测值
        for (int i = 0; i < output_size; i++) {
            predictions[i] = 0.0f;
        }

        // 对每个样本进行预测
        for (int i = 0; i < n_samples; i++) {
            float* sample_preds = &predictions[i * model->config.num_class];
            predict_multiclass(model, &X[i * n_features], sample_preds, model->config.num_class);
            
            // 只对第一个样本打印调试信息
            if (i == 0) {
                printf("DEBUG - First sample prediction process:\n");
                printf("Total trees: %d, num_class: %d\n", model->n_trees, model->config.num_class);
                
                // 详细显示每个类别的预测过程
                for (int c = 0; c < model->config.num_class; c++) {
                    float raw_score = 0.0f;
                    int trees_per_class = model->n_trees / model->config.num_class;
                    
                    for (int t = 0; t < trees_per_class; t++) {
                        int tree_index = t * model->config.num_class + c;
                        if (tree_index < model->n_trees && model->trees && model->trees[tree_index]) {
                            float tree_pred = 0.0f;
                            predict_tree(model->trees[tree_index], &X[i * n_features], &tree_pred, TASK_REGRESSION);
                            raw_score += tree_pred;
                            if (t < 3) { // 只显示前3棵树
                                printf("  Class %d, Tree %d (index %d): pred = %f\n", c, t, tree_index, tree_pred);
                            }
                        }
                    }
                    printf("  Class %d total raw score: %f\n", c, raw_score);
                }
                
                for (int c = 0; c < model->config.num_class; c++) {
                    printf("Class %d probability: %f\n", c, sample_preds[c]);
                }
            }
        }

        // 打印前5个样本的预测值
        printf("Final predictions (first 5 samples):\n");
        for (int i = 0; i < 5 && i < n_samples; i++) {
            printf("Sample %d: ", i);
            for (int c = 0; c < model->config.num_class; c++) {
                printf("%.4f ", predictions[i * model->config.num_class + c]);
            }
            printf("\n");
        }
    } else {
        // 二分类和回归预测
    // 初始化预测值
    for (int i = 0; i < n_samples; i++) {
        predictions[i] = 0.0f;
    }

    // 对每棵树进行预测
    for (int t = 0; t < model->n_trees; t++) {
            // 为每个样本进行预测
            for (int i = 0; i < n_samples; i++) {
                float tree_pred = 0.0f;
                // 获取当前样本的特征
                float* sample_features = &X[i * n_features];
                predict_tree(model->trees[t], sample_features, &tree_pred, model->config.task_type);
                predictions[i] += model->config.learning_rate * tree_pred;
        }

        // 打印前5棵树的预测值
        if (t < 5) {
            printf("Tree %d predictions (first 5 samples):\n", t);
            for (int i = 0; i < 5 && i < n_samples; i++) {
                    printf("Sample %d: %f\n", i, predictions[i]);
        }
            }
    }

    // 打印最终预测值
    printf("Final predictions (first 5 samples):\n");
    for (int i = 0; i < 5 && i < n_samples; i++) {
        printf("Sample %d: %f\n", i, predictions[i]);
        }

        // 对于二分类任务，应用sigmoid变换
        if (model->config.task_type == TASK_BINARY) {
            for (int i = 0; i < n_samples; i++) {
                predictions[i] = 1.0f / (1.0f + expf(-predictions[i]));
            }
            printf("After sigmoid transform (first 5 samples):\n");
            for (int i = 0; i < 5 && i < n_samples; i++) {
                printf("Sample %d: %f\n", i, predictions[i]);
            }
        }
    }

    return predictions;
}

// 修改训练函数
int train(XGBoostModel* model, float* X, float* y, int n_samples, int n_features) {
    // 打印训练配置
    printf("Training configuration:\n");
    printf("Task type: %d\n", model->config.task_type);
    printf("Number of trees: %d\n", model->n_trees);
    printf("Learning rate: %f\n", model->config.learning_rate);
    printf("Max depth: %d\n", model->config.max_depth);
    printf("Min samples split: %d\n", model->config.min_samples_split);
    printf("Lambda: %f\n", model->config.lambda_param);
    printf("Number of features: %d\n", n_features);
    printf("Number of samples: %d\n", n_samples);

    if (model->config.task_type == TASK_MULTICLASS) {
        // 多分类训练 - 使用优化的方法
        printf("Starting multiclass training with %d classes\n", model->config.num_class);
        
        // 为每个类别创建预测数组
        float** class_predictions = (float**)malloc(model->config.num_class * sizeof(float*));
        for (int c = 0; c < model->config.num_class; c++) {
            class_predictions[c] = (float*)calloc(n_samples, sizeof(float));
        }

        // 预计算每个样本的类别标签（避免重复计算）
        int* y_int = (int*)malloc(n_samples * sizeof(int));
        for (int i = 0; i < n_samples; i++) {
            y_int[i] = (int)y[i];
        }

        // 训练每棵树（为每个类别训练）
        int total_trees = model->config.n_estimators * model->config.num_class;
        int trees_completed = 0;
        
        for (int t = 0; t < model->config.n_estimators; t++) {
            if (t % 10 == 0) {
                printf("\nTraining iteration %d/%d (%.1f%% complete)\n", 
                       t + 1, model->config.n_estimators, 
                       (float)(t + 1) / model->config.n_estimators * 100);
            }
            
            // 为每个类别训练一棵树
            for (int c = 0; c < model->config.num_class; c++) {
                // 计算当前类别的梯度 - 使用优化的方法
                float* gradients = (float*)malloc(n_samples * sizeof(float));
                float* hessians = (float*)malloc(n_samples * sizeof(float));
                if (!gradients || !hessians) {
                    printf("Error: Failed to allocate memory for gradients/hessians\n");
                    for (int i = 0; i < model->config.num_class; i++) {
                        free(class_predictions[i]);
                    }
                    free(class_predictions);
                    free(y_int);
                    if (gradients) free(gradients);
                    if (hessians) free(hessians);
                    return -1;
                }

                // 计算梯度和二阶导数 - 使用优化的方法
                for (int i = 0; i < n_samples; i++) {
                    // 当前类别的预测分数
                    float score = class_predictions[c][i];
                    
                    // 当前样本是否属于当前类别
                    float true_val = (y_int[i] == c) ? 1.0f : 0.0f;
                    
                    // 使用稳定的梯度计算
                    gradients[i] = (score - true_val);
                    hessians[i] = 1.0f; // 简化的二阶导数
                }

                // 创建新树
                TreeNode* new_tree = create_tree_node(-1, 0.0f, 0.0f, 0);
                if (!new_tree) {
                    printf("Error: Failed to create tree for class %d\n", c);
                    for (int i = 0; i < model->config.num_class; i++) {
                        free(class_predictions[i]);
                    }
                    free(class_predictions);
                    free(y_int);
                    free(gradients);
                    free(hessians);
                    return -1;
                }

                // 构建树
                split_tree_node(new_tree, X, gradients, hessians,
                              n_samples, n_features, 0, &model->config);
                
                // 更新预测值
                for (int i = 0; i < n_samples; i++) {
                    float tree_pred = 0.0f;
                    float* sample_features = &X[i * n_features];
                    predict_tree(new_tree, sample_features, &tree_pred, TASK_REGRESSION);
                    class_predictions[c][i] += model->config.learning_rate * tree_pred;
                }

                // 添加树到模型
                model->trees[model->n_trees++] = new_tree;
                trees_completed++;
                
                // 显示进度
                if (trees_completed % 50 == 0) {
                    printf("Completed %d/%d trees (%.1f%%)\n", 
                           trees_completed, total_trees,
                           (float)trees_completed / total_trees * 100);
                }
                
                free(gradients);
                free(hessians);
            }
        }

        // 清理内存
        for (int c = 0; c < model->config.num_class; c++) {
            free(class_predictions[c]);
        }
        free(class_predictions);
        free(y_int);
        
    } else {
        // 二分类和回归训练
    float* predictions = (float*)malloc(n_samples * sizeof(float));
    if (!predictions) {
        printf("Error: Failed to allocate memory for predictions\n");
        return -1;
    }

    // 初始化预测值
    for (int i = 0; i < n_samples; i++) {
        predictions[i] = 0.0f;
    }

    // 训练每棵树
    for (int t = 0; t < model->config.n_estimators; t++) {
            if (t % 10 == 0) {
                printf("\nTraining tree %d/%d (%.1f%% complete)\n", 
                       t + 1, model->config.n_estimators,
                       (float)(t + 1) / model->config.n_estimators * 100);
            }
        
        // 计算梯度
        float* gradients = (float*)malloc(n_samples * sizeof(float));
        float* hessians = (float*)malloc(n_samples * sizeof(float));
        if (!gradients || !hessians) {
            printf("Error: Failed to allocate memory for gradients/hessians\n");
            free(predictions);
            if (gradients) free(gradients);
            if (hessians) free(hessians);
            return -1;
        }

        // 计算梯度和二阶导数
        for (int i = 0; i < n_samples; i++) {
            float pred = predictions[i];
            float true_val = y[i];
            
                // 使用正确的任务类型进行梯度计算
                gradients[i] = calculate_gradient(true_val, pred, model->config.task_type);
                hessians[i] = calculate_hessian(true_val, pred, model->config.task_type);
        }

        // 创建新树
        TreeNode* new_tree = create_tree_node(-1, 0.0f, 0.0f, 0);
        if (!new_tree) {
            printf("Error: Failed to create tree %d\n", t);
            free(predictions);
            free(gradients);
            free(hessians);
            return -1;
        }

        // 构建树
        split_tree_node(new_tree, X, gradients, hessians,
                          n_samples, n_features, 0, &model->config);
        
        // 更新预测值
            for (int i = 0; i < n_samples; i++) {
                float tree_pred = 0.0f;
                // 获取当前样本的特征
                float* sample_features = &X[i * n_features];
                predict_tree(new_tree, sample_features, &tree_pred, model->config.task_type);
                predictions[i] += model->config.learning_rate * tree_pred;
        }
        
        // 添加树到模型
        model->trees[model->n_trees++] = new_tree;
        
            free(gradients);
            free(hessians);
        }

        free(predictions);
    }

    printf("\nTraining completed successfully!\n");
    printf("Total trees created: %d\n", model->n_trees);
    return 0;
}

// Free tree nodes recursively
static void free_tree_nodes(TreeNode* node) {
    if (!node) return;

    free_tree_nodes(node->left);
    free_tree_nodes(node->right);
    free(node);
}

// Free XGBoost model
void free_xgboost_model(XGBoostModel* model) {
    if (!model) return;

    if (model->trees) {
        for (int i = 0; i < model->n_trees; i++) {
            if (model->trees[i]) {
                free_tree_nodes(model->trees[i]);
                model->trees[i] = NULL;
            }
        }
        free(model->trees);
        model->trees = NULL;
    }

    if (model->context) {
        aclrtDestroyContext(model->context);
        model->context = NULL;
        aclrtResetDevice(0);
        aclFinalize();
    }

    free(model);
    printf("Model resources freed successfully\n");
}


