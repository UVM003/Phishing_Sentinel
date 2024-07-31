# Step 1 import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import joblib


legitimate_df = pd.read_csv("structured_data_legitimate_final.csv")
phishing_df = pd.read_csv("ostructured_data_phishing_final.csv")


df = pd.concat([legitimate_df, phishing_df], axis=0)

df = df.sample(frac=1)
print(df.shape)


df = df.drop('URL', axis=1)

df = df.drop_duplicates()

X = df.drop('label', axis=1)
Y = df['label']
print(X.shape)
print(Y.shape)


x_train1, x_test1, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
scaler=StandardScaler()
x_train = scaler.fit_transform(x_train1)
x_test = scaler.transform(x_test1)

rf_model = RandomForestClassifier(random_state=42,n_estimators=60)

dt_model = tree.DecisionTreeClassifier()


xgb_model = XGBClassifier(subsample= 1.0, n_estimators= 200, max_depth= 6,learning_rate= 0.1, colsample_bytree= 0.8)


K = 7
total = X.shape[0]
index = int(total / K)

# 1
X_1_test = X.iloc[:index]
X_1_train = X.iloc[index:]
Y_1_test = Y.iloc[:index]
Y_1_train = Y.iloc[index:]

# 2
X_2_test = X.iloc[index:index*2]
X_2_train = X.iloc[np.r_[:index, index*2:]]
Y_2_test = Y.iloc[index:index*2]
Y_2_train = Y.iloc[np.r_[:index, index*2:]]

# 3
X_3_test = X.iloc[index*2:index*3]
X_3_train = X.iloc[np.r_[:index*2, index*3:]]
Y_3_test = Y.iloc[index*2:index*3]
Y_3_train = Y.iloc[np.r_[:index*2, index*3:]]

# 4
X_4_test = X.iloc[index*3:index*4]
X_4_train = X.iloc[np.r_[:index*3, index*4:]]
Y_4_test = Y.iloc[index*3:index*4]
Y_4_train = Y.iloc[np.r_[:index*3, index*4:]]

# 5
X_5_test = X.iloc[index*4:index*5]
X_5_train = X.iloc[np.r_[:index*4,index*5:]]
Y_5_test = Y.iloc[index*4:index*5]
Y_5_train = Y.iloc[np.r_[:index*4,index*5:]]

# 6
X_6_test = X.iloc[index*5:index*6]
X_6_train = X.iloc[np.r_[:index*5,index*6:]]
Y_6_test = Y.iloc[index*5:index*6]
Y_6_train = Y.iloc[np.r_[:index*5,index*6:]]

# 7
X_7_test = X.iloc[index*6:]
X_7_train = X.iloc[:index*6]
Y_7_test = Y.iloc[index*6:]
Y_7_train = Y.iloc[:index*6]

# X and Y train and test lists
X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train,X_6_train,X_7_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test,X_6_test,X_7_test]

Y_train_list = [Y_1_train, Y_2_train, Y_3_train, Y_4_train, Y_5_train, Y_6_train, Y_7_train]
Y_test_list = [Y_1_test, Y_2_test, Y_3_test, Y_4_test, Y_5_test,Y_6_test,Y_7_test]



def calculate_measures(TN, TP, FN, FP):
    total = TP + TN + FN + FP

    if total == 0:
        model_accuracy = float('nan')
    else:
        model_accuracy = (TP + TN) / total

    if (TP + FP) == 0:
        model_precision = float('nan')
    else:
        model_precision = TP / (TP + FP)

    if (TP + FN) == 0:
        model_recall = float('nan')
    else:
        model_recall = TP / (TP + FN)

    return model_accuracy, model_precision, model_recall



rf_accuracy_list, rf_precision_list, rf_recall_list = [], [], []
dt_accuracy_list, dt_precision_list, dt_recall_list = [], [], []
xgb_accuracy_list, xgb_precision_list, xgb_recall_list = [], [], []

for i in range(0, K):
    # ----- RANDOM FOREST ----- #
    rf_model.fit(X_train_list[i], Y_train_list[i])
    rf_predictions = rf_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=rf_predictions).ravel()
    rf_accuracy, rf_precision, rf_recall = calculate_measures(tn, tp, fn, fp)
    rf_accuracy_list.append(rf_accuracy)
    rf_precision_list.append(rf_precision)
    rf_recall_list.append(rf_recall)

    # ----- DECISION TREE ----- #
    dt_model.fit(X_train_list[i], Y_train_list[i])
    dt_predictions = dt_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=dt_predictions).ravel()
    dt_accuracy, dt_precision, dt_recall = calculate_measures(tn, tp, fn, fp)
    dt_accuracy_list.append(dt_accuracy)
    dt_precision_list.append(dt_precision)
    dt_recall_list.append(dt_recall)

  
    # ----- XGB CLASSIFIER ----- #
    xgb_model.fit(X_train_list[i], Y_train_list[i])
    xgb_predictions = xgb_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=xgb_predictions).ravel()
    xgb_accuracy, xgb_precision, xgb_recall = calculate_measures(tn, tp, fn, fp)
    xgb_accuracy_list.append(xgb_accuracy)
    xgb_precision_list.append(xgb_precision)
    xgb_recall_list.append(xgb_recall)


RF_accuracy = sum(rf_accuracy_list) / len(rf_accuracy_list)
RF_precision = sum(rf_precision_list) / len(rf_precision_list)
RF_recall = sum(rf_recall_list) / len(rf_recall_list)

print("Random Forest accuracy ==> ", RF_accuracy)
print("Random Forest precision ==> ", RF_precision)
print("Random Forest recall ==> ", RF_recall)


DT_accuracy = sum(dt_accuracy_list) / len(dt_accuracy_list)
DT_precision = sum(dt_precision_list) / len(dt_precision_list)
DT_recall = sum(dt_recall_list) / len(dt_recall_list)

print("Decision Tree accuracy ==> ", DT_accuracy)
print("Decision Tree precision ==> ", DT_precision)
print("Decision Tree recall ==> ", DT_recall)


xgb_accuracy = sum(xgb_accuracy_list) / len(xgb_accuracy_list)
xgb_precision = sum(xgb_precision_list) / len(xgb_precision_list)
xgb_recall = sum(xgb_recall_list) / len(xgb_recall_list)

print("XGB Classifier accuracy ==> ", xgb_accuracy)
print("XGB Classifier precision ==> ", xgb_precision)
print("XGB Classifier recall ==> ", xgb_recall)

data = {'accuracy': [ DT_accuracy, RF_accuracy,xgb_accuracy],
        'precision': [ DT_precision, RF_precision, xgb_precision],
        'recall': [DT_recall, RF_recall,xgb_recall]
        }

index = [ 'DT', 'RF', 'XGB']

df_results = pd.DataFrame(data=data, index=index)

ax = df_results.plot.bar(rot=0)
plt.show()




def plot_metrics(model_name, accuracy_list, precision_list, recall_list):
    x_labels = [f'Fold {i+1}' for i in range(len(accuracy_list))]
    x = np.arange(len(accuracy_list)) 
    width = 0.2 

    fig, ax = plt.subplots()
    
   
    rects1 = ax.bar(x - width, accuracy_list, width, label='Accuracy')
    rects2 = ax.bar(x, precision_list, width, label='Precision')
    rects3 = ax.bar(x + width, recall_list, width, label='Recall')

   
    ax.set_xlabel('K-Fold Iterations')
    ax.set_ylabel('Scores')
    ax.set_title(f'{model_name} Metrics by K-Fold Iteration')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    fig.tight_layout()
    plt.ylim(0, 1)  
    plt.show()

plot_metrics('Random Forest', rf_accuracy_list, rf_precision_list, rf_recall_list)
plot_metrics('Decision Tree', rf_accuracy_list, rf_precision_list, rf_recall_list)
plot_metrics('XGB Classifier', rf_accuracy_list, rf_precision_list, rf_recall_list)

joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(dt_model, 'dt_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')































