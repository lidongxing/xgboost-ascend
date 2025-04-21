#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include "acl/acl.h"  // Ascend NPU API
#include "xgboost_core.h"

// Predict using a single tree
float predict_tree(TreeNode* tree, float* X, int n_features) {
    if (!tree || !X) {
        printf("Invalid input to predict_tree\n");
        return 0.0f;
    }
    
    TreeNode* node = tree;
    while (node->feature_index != -1) {
        if (node->feature_index >= n_features) {
            printf("Invalid feature index: %d\n", node->feature_index);
            return 0.0f;
        }
        
        if (X[node->feature_index] <= node->split_value) {
            if (!node->left) {
                printf("Left child is NULL\n");
                return 0.0f;
            }
            node = node->left;
        } else {
            if (!node->right) {
                printf("Right child is NULL\n");
                return 0.0f;
            }
            node = node->right;
        }
    }
    
    // 直接返回叶节点的预测值，不进行任何转换
    return node->prediction;
}

// Initialize ACL and create context
static int init_acl(XGBoostModel* model) {
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
        aclFinalize();
        return -1;
    }
    printf("Device set successfully\n");
    
    ret = aclrtCreateContext(&model->context, 0);
    if (ret != ACL_SUCCESS) {
        printf("Failed to create context, error code: %d\n", ret);
        aclrtResetDevice(0);
        aclFinalize();
        return -1;
    }
    printf("Context created successfully\n");
    
    return 0;
}

// Create a tree node
TreeNode* create_tree_node(int feature_index, float split_value, float prediction) {
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    if (!node) {
        printf("Failed to allocate memory for tree node\n");
        return NULL;
    }
    
    node->feature_index = feature_index;
    node->split_value = split_value;
    node->prediction = prediction;
    node->left = NULL;
    node->right = NULL;
    
    return node;
}

// Split tree node
int split_tree_node(TreeNode* node, float* X, float* y, float* weights, int n_samples, int n_features, 
                   int depth, XGBoostConfig* config) {
    if (!node || !X || !y || !weights || n_samples <= 0 || n_features <= 0) {
        printf("Invalid input parameters\n");
        return -1;
    }
    
    // 检查是否达到最大深度或样本数太少
    if (depth >= config->max_depth || n_samples < config->min_samples_split) {
        node->feature_index = -1;  // 标记为叶节点
        node->split_value = 0.0f;
        
        // 计算叶节点的预测值（使用加权平均）
        float sum_weights = 0.0f;
        float sum_weighted_y = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            sum_weights += weights[i];
            sum_weighted_y += weights[i] * y[i];
        }
        
        if (sum_weights > 0.0f) {
            // 直接使用加权平均作为预测值，不进行任何转换
            node->prediction = sum_weighted_y / sum_weights;
        } else {
            node->prediction = 0.0f;
        }
        
        return 0;
    }
    
    // 寻找最佳分割点
    float best_gain = -1.0f;
    int best_feature = -1;
    float best_value = 0.0f;
    
    for (int f = 0; f < n_features; f++) {
        // 对每个特征值进行排序
        float* feature_values = (float*)malloc(n_samples * sizeof(float));
        int* indices = (int*)malloc(n_samples * sizeof(int));
        
        for (int i = 0; i < n_samples; i++) {
            feature_values[i] = X[i * n_features + f];
            indices[i] = i;
        }
        
        // 简单的冒泡排序
        for (int i = 0; i < n_samples - 1; i++) {
            for (int j = 0; j < n_samples - i - 1; j++) {
                if (feature_values[j] > feature_values[j + 1]) {
                    float temp_val = feature_values[j];
                    feature_values[j] = feature_values[j + 1];
                    feature_values[j + 1] = temp_val;
                    
                    int temp_idx = indices[j];
                    indices[j] = indices[j + 1];
                    indices[j + 1] = temp_idx;
                }
            }
        }
        
        // 尝试每个可能的分割点
        for (int i = 0; i < n_samples - 1; i++) {
            if (feature_values[i] == feature_values[i + 1]) continue;
            
            float split_value = (feature_values[i] + feature_values[i + 1]) / 2.0f;
            
            // 计算分割后的加权平方误差
            float left_sum = 0.0f, right_sum = 0.0f;
            float left_weight = 0.0f, right_weight = 0.0f;
            
            for (int j = 0; j <= i; j++) {
                left_sum += weights[indices[j]] * y[indices[j]];
                left_weight += weights[indices[j]];
            }
            
            for (int j = i + 1; j < n_samples; j++) {
                right_sum += weights[indices[j]] * y[indices[j]];
                right_weight += weights[indices[j]];
            }
            
            float left_mean = (left_weight > 0.0f) ? left_sum / left_weight : 0.0f;
            float right_mean = (right_weight > 0.0f) ? right_sum / right_weight : 0.0f;
            
            // 使用信息增益作为分割标准
            float gain = 0.0f;
            for (int j = 0; j < n_samples; j++) {
                float pred = (j <= i) ? left_mean : right_mean;
                float p = 1.0f / (1.0f + expf(-pred));
                if (p < 0.0001f) p = 0.0001f;
                if (p > 0.9999f) p = 0.9999f;
                gain += weights[j] * (y[j] * logf(p) + (1.0f - y[j]) * logf(1.0f - p));
            }
            
            // 添加正则化项
            float reg_term = 0.5f * (left_weight + right_weight) / n_samples;
            gain += reg_term;
            
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_value = split_value;
            }
        }
        
        free(feature_values);
        free(indices);
    }
    
    // 如果没有找到好的分割点，创建叶节点
    if (best_gain <= 0.0f) {
        node->feature_index = -1;
        node->split_value = 0.0f;
        
        float sum_weights = 0.0f;
        float sum_weighted_y = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            sum_weights += weights[i];
            sum_weighted_y += weights[i] * y[i];
        }
        
        if (sum_weights > 0.0f) {
            // 使用加权平均作为预测值，并应用logit转换
            float p = sum_weighted_y / sum_weights;
            p = fmaxf(0.0001f, fminf(0.9999f, p));
            node->prediction = logf(p / (1.0f - p));
        } else {
            node->prediction = 0.0f;
        }
        
        return 0;
    }
    
    // 创建分割节点
    node->feature_index = best_feature;
    node->split_value = best_value;
    node->prediction = 0.0f;
    
    // 分割数据
    int left_size = 0, right_size = 0;
    for (int i = 0; i < n_samples; i++) {
        if (X[i * n_features + best_feature] <= best_value) {
            left_size++;
        } else {
            right_size++;
        }
    }
    
    float* left_X = (float*)malloc(left_size * n_features * sizeof(float));
    float* left_y = (float*)malloc(left_size * sizeof(float));
    float* left_weights = (float*)malloc(left_size * sizeof(float));
    float* right_X = (float*)malloc(right_size * n_features * sizeof(float));
    float* right_y = (float*)malloc(right_size * sizeof(float));
    float* right_weights = (float*)malloc(right_size * sizeof(float));
    
    int left_idx = 0, right_idx = 0;
    for (int i = 0; i < n_samples; i++) {
        if (X[i * n_features + best_feature] <= best_value) {
            memcpy(left_X + left_idx * n_features, X + i * n_features, n_features * sizeof(float));
            left_y[left_idx] = y[i];
            left_weights[left_idx] = weights[i];
            left_idx++;
        } else {
            memcpy(right_X + right_idx * n_features, X + i * n_features, n_features * sizeof(float));
            right_y[right_idx] = y[i];
            right_weights[right_idx] = weights[i];
            right_idx++;
        }
    }
    
    // 递归构建左右子树
    node->left = create_tree_node(-1, 0.0f, 0.0f);
    node->right = create_tree_node(-1, 0.0f, 0.0f);
    
    if (split_tree_node(node->left, left_X, left_y, left_weights, left_size, n_features, depth + 1, config) != 0 ||
        split_tree_node(node->right, right_X, right_y, right_weights, right_size, n_features, depth + 1, config) != 0) {
        free(left_X);
        free(left_y);
        free(left_weights);
        free(right_X);
        free(right_y);
        free(right_weights);
        return -1;
    }
    
    free(left_X);
    free(left_y);
    free(left_weights);
    free(right_X);
    free(right_y);
    free(right_weights);
    
    return 0;
}

// Create XGBoost model
XGBoostModel* create_xgboost_model(XGBoostConfig config) {
    printf("Creating XGBoost model...\n");
    printf("Config: max_depth=%d, min_samples_split=%d, learning_rate=%f, n_estimators=%d\n",
           config.max_depth, config.min_samples_split, config.learning_rate, config.n_estimators);
    
    XGBoostModel* model = (XGBoostModel*)malloc(sizeof(XGBoostModel));
    if (!model) {
        printf("Failed to allocate memory for XGBoost model\n");
        return NULL;
    }
    
    printf("Initializing model configuration...\n");
    model->config = config;
    model->trees = NULL;
    model->n_trees = 0;
    model->context = NULL;
    
    printf("XGBoost model created successfully\n");
    return model;
}

// Train XGBoost model
int train_xgboost_model(XGBoostModel* model, float* X, float* y, int n_samples, int n_features) {
    if (init_acl(model) != 0) {
        return -1;
    }
    
    model->trees = (TreeNode**)malloc(model->config.n_estimators * sizeof(TreeNode*));
    if (!model->trees) {
        printf("Failed to allocate memory for trees\n");
        return -1;
    }
    
    model->n_trees = 0;
    
    // 初始化样本权重
    float* weights = (float*)malloc(n_samples * sizeof(float));
    if (!weights) {
        printf("Failed to allocate memory for weights\n");
        return -1;
    }
    
    for (int i = 0; i < n_samples; i++) {
        weights[i] = 1.0f / n_samples;  // 初始权重相等
    }
    
    // 训练每棵树
    for (int t = 0; t < model->config.n_estimators; t++) {
        printf("Training tree %d\n", t);
        
        // 计算当前预测值
        float* current_pred = (float*)calloc(n_samples, sizeof(float));
        if (!current_pred) {
            printf("Failed to allocate memory for current predictions\n");
            free(weights);
            return -1;
        }
        
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < t; j++) {
                if (model->trees[j]) {
                    current_pred[i] += model->config.learning_rate * 
                                     predict_tree(model->trees[j], X + i * n_features, n_features);
                }
            }
        }
        
        // 计算残差
        float* residuals = (float*)malloc(n_samples * sizeof(float));
        if (!residuals) {
            printf("Failed to allocate memory for residuals\n");
            free(current_pred);
            free(weights);
            return -1;
        }
        
        for (int i = 0; i < n_samples; i++) {
            // 使用交叉熵损失函数的梯度作为残差
            float p = 1.0f / (1.0f + expf(-current_pred[i]));
            // 添加梯度缩放因子
            residuals[i] = (y[i] - p) * p * (1.0f - p);
        }
        
        // 训练新树
        model->trees[t] = create_tree_node(-1, 0.0f, 0.0f);
        if (split_tree_node(model->trees[t], X, residuals, weights, n_samples, n_features, 0, &model->config) != 0) {
            printf("Failed to train tree %d\n", t);
            free(residuals);
            free(current_pred);
            free(weights);
            return -1;
        }
        
        model->n_trees++;
        
        // 更新样本权重（基于残差的绝对值）
        float total_weight = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            weights[i] = fabsf(residuals[i]);
            total_weight += weights[i];
        }
        
        // 归一化权重
        if (total_weight > 0.0f) {
            for (int i = 0; i < n_samples; i++) {
                weights[i] /= total_weight;
            }
        }
        
        free(residuals);
        free(current_pred);
    }
    
    free(weights);
    return 0;
}

// Predict using XGBoost model
float* predict_xgboost_model(XGBoostModel* model, float* X, int n_samples) {
    printf("Starting prediction in C...\n");
    printf("Model address: %p\n", model);
    printf("Input data address: %p\n", X);
    printf("Number of samples: %d\n", n_samples);
    
    if (!model || !X || n_samples <= 0) {
        printf("Invalid input parameters\n");
        return NULL;
    }
    
    printf("Model configuration:\n");
    printf("  max_depth: %d\n", model->config.max_depth);
    printf("  min_samples_split: %d\n", model->config.min_samples_split);
    printf("  learning_rate: %f\n", model->config.learning_rate);
    printf("  n_estimators: %d\n", model->config.n_estimators);
    printf("  n_trees: %d\n", model->n_trees);
    
    if (!model->trees) {
        printf("No trees in the model\n");
        return NULL;
    }
    
    float* predictions = (float*)malloc(n_samples * sizeof(float));
    if (!predictions) {
        printf("Failed to allocate memory for predictions\n");
        return NULL;
    }
    
    printf("Allocated memory for predictions at %p\n", predictions);
    
    // 假设特征数量是固定的，这里使用10作为特征数量
    int n_features = 10;
    
    for (int i = 0; i < n_samples; i++) {
        printf("Processing sample %d\n", i);
        float prediction = 0.0f;
        int valid_trees = 0;
        
        for (int t = 0; t < model->n_trees; t++) {
            printf("Processing tree %d\n", t);
            if (!model->trees[t]) {
                printf("Tree %d is NULL\n", t);
                continue;
            }
            
            float tree_pred = predict_tree(model->trees[t], X + i * n_features, n_features);
            if (!isnan(tree_pred) && !isinf(tree_pred)) {
                // 使用学习率缩放每个树的预测
                prediction += model->config.learning_rate * tree_pred;
                valid_trees++;
                printf("Tree %d prediction: %f\n", t, tree_pred);
            } else {
                printf("Tree %d produced invalid prediction: %f\n", t, tree_pred);
            }
        }
        
        // 如果至少有一个有效的树预测，则应用sigmoid函数
        if (valid_trees > 0) {
            // 使用sigmoid函数将预测值转换到[0,1]区间
            prediction = 1.0f / (1.0f + expf(-prediction));
            // 确保预测值在合理范围内
            prediction = fmaxf(0.0001f, fminf(0.9999f, prediction));
        } else {
            prediction = 0.5f;  // 默认值
        }
        
        predictions[i] = prediction;
        printf("Sample %d final prediction: %f (from %d valid trees)\n", i, prediction, valid_trees);
    }
    
    printf("Prediction completed successfully\n");
    return predictions;
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
    if (model->trees) {
        for (int i = 0; i < model->n_trees; i++) {
            // Free tree nodes recursively
            free_tree_nodes(model->trees[i]);
        }
        free(model->trees);
    }
    
    aclrtDestroyContext(model->context);
    aclrtResetDevice(0);
    aclFinalize();
}

// Predict function
float* predict(XGBoostModel* model, float* X, int n_samples, int n_features) {
    printf("Starting prediction in C...\n");
    printf("Model address: %p\n", model);
    printf("Input data address: %p\n", X);
    printf("Number of samples: %d\n", n_samples);
    printf("Number of features: %d\n", n_features);
    
    if (!model || !X || n_samples <= 0 || n_features <= 0) {
        printf("Invalid input parameters\n");
        return NULL;
    }
    
    printf("Model configuration:\n");
    printf("  max_depth: %d\n", model->config.max_depth);
    printf("  min_samples_split: %d\n", model->config.min_samples_split);
    printf("  learning_rate: %f\n", model->config.learning_rate);
    printf("  n_estimators: %d\n", model->config.n_estimators);
    printf("  n_trees: %d\n", model->n_trees);
    
    if (!model->trees) {
        printf("No trees in the model\n");
        return NULL;
    }
    
    float* predictions = (float*)malloc(n_samples * sizeof(float));
    if (!predictions) {
        printf("Failed to allocate memory for predictions\n");
        return NULL;
    }
    
    printf("Allocated memory for predictions at %p\n", predictions);
    
    for (int i = 0; i < n_samples; i++) {
        printf("Processing sample %d\n", i);
        float prediction = 0.0f;
        for (int t = 0; t < model->n_trees; t++) {
            printf("Processing tree %d\n", t);
            if (!model->trees[t]) {
                printf("Tree %d is NULL\n", t);
                continue;
            }
            float tree_pred = predict_tree(model->trees[t], X + i * n_features, n_features);
            printf("Tree %d prediction: %f\n", t, tree_pred);
            prediction += tree_pred;
        }
        predictions[i] = prediction;
        printf("Sample %d final prediction: %f\n", i, prediction);
    }
    
    printf("Prediction completed successfully\n");
    return predictions;
}