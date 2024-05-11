import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

df = pd.read_csv("mail_data.csv")

df.fillna("", inplace=True)

df["Category"] = df["Category"].replace({"spam": 0, "ham": 1})


x = df["Message"]
y = df["Category"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

y_train = y_train.astype("int")
y_test = y_test.astype("int")

model = LogisticRegression()
model.fit(x_train_features, y_train)

prediction_train = model.predict(x_train_features)
prediction_test = model.predict(x_test_features)

accuracy_train = accuracy_score(y_train, prediction_train)
accuracy_test = accuracy_score(y_test, prediction_test)

print("Accuracy on test data: ", accuracy_test)
print("Accuracy on train data: ", accuracy_train)

conf_matrix = confusion_matrix(y_test, prediction_test)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False,
            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test_features)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()