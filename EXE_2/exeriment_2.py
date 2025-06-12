import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, learning_curve, 
                                     validation_curve, GridSearchCV, cross_val_score, 
                                     StratifiedKFold, ParameterGrid)  # 新增ParameterGrid
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns
from tqdm import tqdm  # 新增进度条库

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建保存路径
current_dir = os.path.dirname(os.path.abspath(__file__))
pic_path = os.path.join(current_dir, "results/pic")
results_path = os.path.join(current_dir, "results/data")
os.makedirs(pic_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

# 加载数据集
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names

# 创建数据预处理管道
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif)),
    ('svm', SVC(probability=True, random_state=42))
])

# 参数调优 - 更精细的网格搜索
param_grid = {
    'feature_selection__k': [10, 15, 20, 25, 30],  # 尝试不同数量的特征
    'svm__C': np.logspace(-2, 2, 5),  # 正则化参数
    'svm__gamma': np.logspace(-3, 1, 5),  # 核系数
    'svm__kernel': ['rbf', 'linear', 'poly'],  # 核函数
    'svm__degree': [2, 3]  # 多项式核的阶数
}




# 使用StratifiedKFold保持类别比例
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 自定义进度条包装器，兼容旧版tqdm
class TqdmWrapper:
    def __init__(self, iterable=None, total=None, desc=None):
        self.total = total
        self.desc = desc
        self.iterable = iterable
        self.pbar = None
        self._create_pbar()
        
    def _create_pbar(self):
        try:
            # 尝试使用完整的tqdm
            from tqdm import tqdm
            self.pbar = tqdm(self.iterable, total=self.total, desc=self.desc)
        except:
            # 如果tqdm不兼容，使用简化版
            class SimpleProgressBar:
                def __init__(self, total, desc):
                    self.total = total
                    self.desc = desc
                    self.current = 0
                    
                def update(self, n=1):
                    self.current += n
                    percent = 100 * self.current / self.total
                    print(f"\r{self.desc}: {percent:.1f}% ({self.current}/{self.total})", end='')
                    
                def close(self, *args, **kwargs):
                    print()
                    
            self.pbar = SimpleProgressBar(self.total, self.desc)
            
    def update(self, n=1):
        self.pbar.update(n)
        
    def close(self):
        self.pbar.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False



# 执行网格搜索（去掉 callback 和自定义进度条）
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,  # 使用自带日志显示进度
    return_train_score=True
)
grid_search.fit(X, y)

best_params = grid_search.best_params_
print("最佳参数:", best_params)

# 获取最佳模型
best_model = grid_search.best_estimator_

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 模型训练进度条（可选，单次fit可以用tqdm包裹）
from tqdm import tqdm
with tqdm(desc="模型训练进度", total=1, unit="epoch") as pbar:
    best_model.fit(X_train, y_train)
    pbar.update(1)




# 保存最佳模型参数
with open(os.path.join(results_path, "best_params.txt"), "w") as f:
    f.write(str(best_params))

# 交叉验证评估
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
print(f"交叉验证准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 拟合模型（添加训练进度条）
with tqdm(desc="模型训练进度", total=1, unit="epoch") as pbar:
    best_model.fit(X_train, y_train)
    pbar.update(1)



# 预测
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 模型评估
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['恶性', '良性'])

# 保存评估结果到文件
with open(os.path.join(results_path, "model_metrics.txt"), "w") as f:
    f.write(f"模型最佳参数: {best_params}\n\n")
    f.write(f"交叉验证准确率: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n\n")
    f.write(f"测试集准确率: {accuracy:.4f}\n\n")
    f.write("混淆矩阵:\n")
    f.write(f"{cm}\n\n")
    f.write("分类报告:\n")
    f.write(report)

# ----------------- 可视化部分 -----------------

# 1. 优化后的学习曲线
train_sizes, train_scores, valid_scores = learning_curve(
    best_model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv, scoring='accuracy', n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.figure(figsize=(12, 7))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, valid_mean, color='green', marker='s', markersize=5, label='Validation Accuracy')
plt.fill_between(train_sizes, valid_mean + valid_std, valid_mean - valid_std, alpha=0.15, color='green')
plt.title('Optimized SVM Learning Curve')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(pic_path, "learning_curve_optimized.png"))
plt.close()


# 2. 优化后的验证曲线 (C参数)
param_range = np.logspace(-2, 2, 5)
train_scores_c, test_scores_c = validation_curve(
    best_model, X_train, y_train, param_name="svm__C", param_range=param_range,
    cv=cv, scoring="accuracy", n_jobs=-1
)

train_mean_c = np.mean(train_scores_c, axis=1)
train_std_c = np.std(train_scores_c, axis=1)
test_mean_c = np.mean(test_scores_c, axis=1)
test_std_c = np.std(test_scores_c, axis=1)

plt.figure(figsize=(12, 7))
plt.plot(param_range, train_mean_c, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(param_range, train_mean_c + train_std_c, train_mean_c - train_std_c, alpha=0.15, color='blue')
plt.plot(param_range, test_mean_c, color='green', marker='s', markersize=5, label='Validation Accuracy')
plt.fill_between(param_range, test_mean_c + test_std_c, test_mean_c - test_std_c, alpha=0.15, color='green')
plt.title('Optimized SVM Validation Curve (C parameter)')
plt.xlabel('C Value')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(pic_path, "validation_curve_c_optimized.png"))
plt.close()

# 3. ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Optimized SVM ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(pic_path, "roc_curve_optimized.png"))
plt.close()

# 4. 混淆矩阵可视化
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.title('Optimized SVM Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(pic_path, "confusion_matrix_optimized.png"))
plt.close()

# 5. 精确率-召回率曲线
precision, recall, _ = precision_recall_curve(y_test, y_prob)
average_precision = average_precision_score(y_test, y_prob)

plt.figure(figsize=(12, 7))
plt.plot(recall, precision, color='blue', lw=2, 
         label=f'Precision-Recall Curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Optimized SVM Precision-Recall Curve')
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig(os.path.join(pic_path, "precision_recall_curve.png"))
plt.close()

# 6. 特征重要性
if hasattr(best_model.named_steps['svm'], 'coef_'):
    if best_model.named_steps['svm'].kernel == 'linear':
        coef = best_model.named_steps['svm'].coef_[0]
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coef
        })
        feature_importance = feature_importance.sort_values('Importance', key=abs, ascending=False)
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance (Linear SVM)')
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.tight_layout()
        plt.savefig(os.path.join(pic_path, "feature_importance_linear.png"))
        plt.close()
        
        feature_importance.to_csv(os.path.join(results_path, "feature_importance.csv"), index=False)
else:
    selector = best_model.named_steps['feature_selection']
    feature_scores = selector.scores_
    indices = np.argsort(feature_scores)[-20:]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.barh(range(len(indices)), feature_scores[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Score')
    plt.tight_layout()
    plt.savefig(os.path.join(pic_path, "feature_importance.png"))
    plt.close()


# 7. PCA可视化
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(best_model.named_steps['scaler'].transform(X_train))

plt.figure(figsize=(12, 7))
plt.scatter(X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1], 
           c='red', marker='o', label='Malignant', alpha=0.7, s=50)
plt.scatter(X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1], 
           c='blue', marker='x', label='Benign', alpha=0.7, s=50)
plt.title('Sample Distribution after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best')
plt.grid(True)
plt.savefig(os.path.join(pic_path, "pca_visualization_optimized.png"))
plt.close()

# 8. 交叉验证结果可视化
cv_results = pd.DataFrame(grid_search.cv_results_)
top_params = cv_results.nlargest(5, 'mean_test_score')

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.title('Cross-Validation Results for Different Parameter Combinations')
plt.plot(cv_results.index, cv_results['mean_test_score'], 'o-', label='Validation Accuracy')
plt.plot(cv_results.index, cv_results['mean_train_score'], 'o-', label='Training Accuracy')
plt.xlabel('Parameter Combination')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
sns.barplot(x='rank_test_score', y='mean_test_score', data=top_params)
for i, v in enumerate(top_params['mean_test_score']):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center')
plt.title('Validation Accuracy of Top 5 Parameter Combinations')
plt.xlabel('Parameter Combination Rank')
plt.ylabel('Mean Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(pic_path, "cv_results.png"))
plt.close()

# 9. 不同阈值下的精确率和召回率
thresholds = np.linspace(0, 1, 21)
precisions = []
recalls = []
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_prob > threshold).astype(int)
    cm_threshold = confusion_matrix(y_test, y_pred_threshold)
    
    # 计算精确率和召回率
    if (cm_threshold[1, 1] + cm_threshold[0, 1]) == 0:
        precision = 1.0
    else:
        precision = cm_threshold[1, 1] / (cm_threshold[1, 1] + cm_threshold[0, 1])
    
    if (cm_threshold[1, 1] + cm_threshold[1, 0]) == 0:
        recall = 1.0
    else:
        recall = cm_threshold[1, 1] / (cm_threshold[1, 1] + cm_threshold[1, 0])
    
    # 计算F1分数
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

plt.figure(figsize=(12, 7))
plt.plot(thresholds, precisions, 'b-', label='Precision')
plt.plot(thresholds, recalls, 'g-', label='Recall')
plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 Score at Different Thresholds')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(pic_path, "threshold_analysis.png"))
plt.close()


print(f"所有图表已保存至: {pic_path}")
print(f"模型评估结果已保存至: {results_path}")