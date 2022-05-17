
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, log_loss, accuracy_score
from get_features import create_final_df


df_new, df_features, labels = create_final_df()

X_train, X_test, y_train, y_test = train_test_split(df_new, labels, test_size=0.2, random_state=42)

#clf = SVC(gamma='auto').fit(X_train, y_train)
#clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100).fit(X_train, y_train)
a = 0.5
s_weight = np.where(y_train == 1, a, 1-a)
clf = LogisticRegression(random_state=1).fit(X_train, y_train, sample_weight=s_weight)
preds = clf.predict(X_test)
print(confusion_matrix(y_test, preds))
print(accuracy_score(y_test, preds))
#print(classification_report(y_test, preds, target_names = ['0', '1']))