from sklearn import datasets, neighbors, cross_validation, grid_search, metrics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

digits= datasets.load_digits()
x= digits.data[:1000]
y=digits.target[:1000]

kf= cross_validation.KFold(len(x), n_folds= 10, shuffle= True)

'''
Some notes about the cross validation used above using KFold:
The first argument above is the # of samples we want to deal with. Since we’re doing KFCV over our entire 1000-sample digits dataset, we set it to the length of x.
The second argument is the # of folds; 10 is often used.
The third argument indicates whether you are returning the indices of the original data rather than the elements themselves. Useful if you need to continue working with the original data.
The fourth argument , shuffle, means that KFCV will mix around the data (as you’ll see below), meaning the generated test data won’t be necessarily 0, 1, 2, 3, etc. Although shuffle allows you to throw in some randomness, you don’t want to set shuffle to true for some datasets. For example, if you’re working with the well-known iris dataset where samples 1-50 are from one kind of flower and samples from 50 on are for another kind of flower, you don’t necessarily want to mix up samples 1-50 and 50+.
The fifth argument controls the degree of randomness.
'''

'''
for train,test in kf:
    print (train,'\n\ntest:', test, '\n\n')
'''

'''
#Printing the shapes of the returned train, testing data from the KFold
print(train.shape)
print(test.shape)
'''

'''
one_score= []
kf_cross_val_score= []
k_list= [i for i in range(1, 50) if(i%2 != 0)]

for i in k_list:
        knn= neighbors.KNeighborsClassifier(n_neighbors= i)
        kf= cross_validation.KFold(len(x), n_folds= 10, shuffle= True, random_state= 4)
        for train_indices, test_indices in kf:
            one_score.append(knn.fit(x[train_indices], y[train_indices]).score(x[test_indices],y[test_indices]))
        one_score_mean= np.mean(one_score)
        kf_cross_val_score.append(one_score_mean)

print(np.mean(kf_cross_val_score))
'''
'''
optimal_k= k_list[kf_cross_val_score.index(max(kf_cross_val_score))]
print(optimal_k)

import matplotlib.pyplot as plt
plt.xticks(k_list)
plt.xlim(0, 25)
plt.plot(k_list, kf_cross_val_score, c= 'k')
plt.scatter(k_list, kf_cross_val_score)
'''

'''
From the above plot we can see that:
The accuracy of the plot declines after k=11
and that any value from 1 (high bias) and
11 (comparitively higher bias) are appropriate.
'''

'''
(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)
'''
'''
#print(knn.get_params)
tuned_params= [{'n_neighbors': [1, 3, 5, 7, 11], 'algorithm': ['ball_tree', 'kd_tree'], 'leaf_size': [(i+10) for i in range(50)], 'weights': ['uniform', 'distance'], 'p': [1, 2]}]
'''
'''
leaf_size parameter has no significant outcome on the events.
'''
'''
#, 'metric_params': [0.5, 0.7, 1, 2]}]
#'leaf_size': [(i+10) for i in range(50)]}]
lrgs = grid_search.GridSearchCV(estimator=knn, param_grid= tuned_params, n_jobs=1)
print(np.mean([lrgs.fit(x[train],y[train]).score(x[test],y[test]) for train, test in kf]))
'''
'''
After autotuning the hyperparameters, the result of the cross_validation_score are calculated
and the results of the mean of the scores are printed.
'''
'''
To view the tabular cv_scores_ of each parameter's change in estimation.
'''
'''
print(pd.DataFrame(lrgs.grid_scores_))
'''
'''       
print(np.mean(kf_cross_val_score))
print(lrgs.best_score_)
print(lrgs.best_estimator_.n_neighbors)
print(lrgs.best_estimator_.algorithm)
print(lrgs.best_estimator_.weights)
print(lrgs.best_estimator_.leaf_size)
print(lrgs.best_estimator_.p)
'''
'''
After using GridSearchCV, we find out that the best hyperparameters are:
    1. k=5
    2. weights= 'distance'
    3. algorithm= 'kd_tree'
    4. leaf_size= 10
'''

knn= neighbors.KNeighborsClassifier(n_neighbors= 5, weights= 'distance', algorithm= 'kd_tree', leaf_size= 10)

'''
So the optimal parameters for the KNearestNeighbors are:
    knn= neighbors.KNeighborsClassifier(n_neighbors= 5, algorithm= 'kd_tree', weights= 'distance', leaf_size= 10)
'''
'''
print(cross_validation.cross_val_score(lrgs, x, y, cv=kf, n_jobs=1))
print(np.mean(cross_validation.cross_val_score(lrgs, x, y, cv=kf, n_jobs=1)))
'''
'''
Hence, we obtain a cross validation score after parameter optimization as:
    approximately 98.6-98.7% which is really good indication to start predicting!
'''
train_arr= []
test_arr= []
for train, test in kf:
    train_arr.append(train)
    train_arr.append(test)
x_train, x_test, y_train, y_test= x[train], x[test], y[train], y[test]
knn= knn.fit(x_train, y_train)

y_pred= knn.predict(x_test)
cm= pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'])
#, margins= True
print(cm)


cm_= cm.apply(lambda r: 100.0*r/r.sum())
print(cm_)
sns.set(font_scale= 1.4)
sns.heatmap(cm, annot=True, annot_kws={"size": 16})
plt.show()

y_pred_prob= knn.predict_proba(x_test)[:, 1].ravel()
print(y_test)
'''
fpr, tpr, thresholds= roc_curve(y_test.ravel(), y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label= 'KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title()
plt.show()
'''
'''
label, count= np.unique(y_test, return_counts= True)
y_pred= knn.predict(x_test)
metrics.confusion_matrix(y_test, y_pred, label)
print(label, count)
'''
