import os
import numpy as np
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import joblib

# ----------- 路径设置 -----------
DATA_PATH = "/data1/JiamingLiu/模式识别/ORL"
MODEL_DIR = "/data1/JiamingLiu/模式识别/Code/Exe_1/Model"
RESULT_DIR = "/data1/JiamingLiu/模式识别/Code/Exe_1/Results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ----------- 1. ORL数据加载 -----------
def load_orl_dataset(data_path, img_size=(112,92)):
    faces = []
    labels = []
    transform = transforms.Compose([
        transforms.Resize(img_size), # 按 ORL原始格式
        transforms.ToTensor(),       # [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    for person_dir in tqdm(sorted(os.listdir(data_path)), desc="Loading ORL dataset"):
        person_path = os.path.join(data_path, person_dir)
        if not os.path.isdir(person_path):
            continue
        for img_name in sorted(os.listdir(person_path)):
            if img_name.lower().endswith('.pgm'):
                img_path = os.path.join(person_path, img_name)
                img = Image.open(img_path)
                if img.mode != 'L':
                    img = img.convert('L')
                img = transform(img) # 1 x H x W, torch
                faces.append(img.numpy().flatten())
                labels.append(person_dir) # s1, s2, ..., s40
    return np.array(faces), np.array(labels)

def evaluate_rf_pca(X, y, n_components, n_estimators, max_depth, test_size=0.3):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42)
    
    # PCA
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # RF
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf.fit(X_train_pca, y_train)
    
    # 训练集和测试集准确率
    train_accuracy = rf.score(X_train_pca, y_train)
    y_pred = rf.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # CV（用训练集）
    cv_scores = cross_val_score(rf, X_train_pca, y_train, cv=5)
    
    class_report = classification_report(
        y_test, y_pred, target_names=le.inverse_transform(np.unique(y_test))
    )
    
    return {
        "pca": pca,
        "rf": rf,
        "le": le,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_scores": cv_scores,
        "mean_cv_accuracy": cv_scores.mean(),
        "classification_report": class_report,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
    }

def plot_learning_curve(rf, X, y, title="Learning Curve", filepath="learning_curve.png"):
    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy"
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(filepath)
    plt.close()

def visualize_results(X_pca, y_true, y_pred, le, pca, max_classes=20):
    labels, counts = np.unique(y_true, return_counts=True)
    if len(labels) > max_classes:
        top_label_indices = np.argsort(counts)[::-1][:max_classes]
        chosen_labels = labels[top_label_indices]
        idx = np.isin(y_true, chosen_labels)
        y_true, y_pred = y_true[idx], y_pred[idx]
        labels = chosen_labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_names = le.inverse_transform(labels)
    plt.figure(figsize=(max(8,max_classes//2), max(6,max_classes//2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
        xticklabels=class_names, yticklabels=class_names,
        vmax=max(cm.max(), 3))
    plt.title(f"Confusion Matrix (Top {len(labels)})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'))
    plt.close()

    # PCA解释方差
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid()
    plt.savefig(os.path.join(RESULT_DIR, 'pca_explained_variance.png'))
    plt.close()

def main():
    print("Loading ORL dataset...")
    t0 = time.time()
    X, y = load_orl_dataset(DATA_PATH, img_size=(112,92))
    print(f"\nDataset loaded in {time.time() - t0:.2f} seconds")
    print(f"Total samples: {len(X)}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # 超参数网格
    n_components_list = [5,10,15,20,25,30,35,40,50,60,70, 80,90,95,100,110,120,130,140,150]
    n_estimators_list = [25,50, 75,100, 150,200,250,300,400,500]
    max_depth_list = [None, 5,10, 15,20,30,40,50]
    
    best_res = None
    best_score = -1
    best_params = {}
    param_grid = []
    print("\nHyperparameter grid search ...")

    for n_components in n_components_list:
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                param_grid.append((n_components, n_estimators, max_depth))
    
    loop = tqdm(param_grid, desc="Grid Search")
    for n_components, n_estimators, max_depth in loop:
        res = evaluate_rf_pca(
            X, y,
            n_components=n_components,
            n_estimators=n_estimators,
            max_depth=max_depth,
            test_size=0.3
        )
        score = res["test_accuracy"]
        loop.set_postfix({
            "n_comp": n_components,
            "n_estim": n_estimators,
            "max_depth": max_depth,
            "train_acc": f"{res['train_accuracy']:.4f}",
            "test_acc": f"{score:.4f}"
        })
        if score > best_score:
            best_score = score
            best_res = res
            best_params = {
                "n_components": n_components,
                "n_estimators": n_estimators,
                "max_depth": max_depth
            }
    
    print("\nBest params found:")
    print(best_params)
    print(f"Best test accuracy: {best_score:.4f}")
    print(f"Corresponding train accuracy: {best_res['train_accuracy']:.4f}")

    # 可视化最优情况
    visualize_results(best_res["X_test_pca"], best_res["y_test"], best_res["y_pred"], best_res["le"], best_res["pca"], max_classes=20)
    
    # 绘制学习曲线
    plot_learning_curve(
        best_res["rf"],
        np.vstack([best_res["X_train_pca"], best_res["X_test_pca"]]),
        np.concatenate([best_res["y_train"], best_res["y_test"]]),
        title="Learning Curve (Random Forest)",
        filepath=os.path.join(RESULT_DIR, 'learning_curve.png')
    )

    # 保存模型
    joblib.dump(best_res["pca"], os.path.join(MODEL_DIR, 'pca_model_orl_best_rf.pkl'))
    joblib.dump(best_res["rf"], os.path.join(MODEL_DIR, 'rf_model_orl_best.pkl'))
    joblib.dump(best_res["le"], os.path.join(MODEL_DIR, 'label_encoder_orl_best.pkl'))
    print("\nBest model saved to disk.")

    # 保存报告
    result_txt = os.path.join(RESULT_DIR, "report_orl_best_rf.txt")
    with open(result_txt, "w", encoding="utf-8") as fout:
        fout.write(f"Best params:\n{best_params}\n\n")
        fout.write(f"Total samples: {len(X)}\n")
        fout.write(f"Number of classes: {len(np.unique(y))}\n")
        fout.write(f"Train accuracy: {best_res['train_accuracy']:.4f}\n")
        fout.write(f"Test accuracy: {best_res['test_accuracy']:.4f}\n")
        fout.write(f"CV scores: {best_res['cv_scores']}\n")
        fout.write(f"Mean CV accuracy: {best_res['mean_cv_accuracy']:.4f}\n\n")
        fout.write("Classification Report:\n")
        fout.write(best_res["classification_report"])
    print(f'Results saved to {result_txt}')

if __name__ == "__main__":
    main()