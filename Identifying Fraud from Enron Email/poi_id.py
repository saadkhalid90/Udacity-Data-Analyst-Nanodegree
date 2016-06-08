##machine learning project
#!/usr/bin/python

## importing modules
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

## function to create two new features; fraction_emails_to_poi and fraction_emails_from_poi
def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if poi_messages == "NaN" or all_messages == "NaN" or all_messages == 0:
        fraction = 0
    else:
        fraction = poi_messages/ float(all_messages)

    return fraction


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## removing outliers (TASK # 2)
data_dict.pop("TOTAL", 0)

## creating two new features using the computeFraction function (TASK # 3)
for index in data_dict.keys():
    data_dict[index]['fraction_emails_from_poi'] = \
    computeFraction(data_dict[index]['from_this_person_to_poi'], \
    data_dict[index]['from_messages'])

    data_dict[index]['fraction_emails_to_poi'] = \
    computeFraction(data_dict[index]['from_poi_to_this_person'], \
    data_dict[index]['to_messages'])


##features_list = list_features - Initially adding many features later to be
##selected by SelectKBest  -- (TASK # 1)

features_list = ['poi', \
                'salary', \
                'fraction_emails_from_poi', \
                'fraction_emails_to_poi', \
                'exercised_stock_options', \
                'total_stock_value', \
                'expenses',\
                'shared_receipt_with_poi',\
                'bonus', \
                'total_payments']


### Store to my_dataset for easy export below.

## commented list of features for easy reference
'''
All features available:
1 to_messages
2 deferral_payments
3 expenses
4 poi
5 deferred_income
6 email_address
7 long_term_incentive
8 fraction_emails_from_poi
9 restricted_stock_deferred
10 shared_receipt_with_poi
11 loan_advances
12 from_messages
13 other
14 director_fees
15 bonus
16 total_stock_value
17 from_poi_to_this_person
18 from_this_person_to_poi
19 restricted_stock
20 salary
21 total_payments
22 fraction_emails_to_poi
23 exercised_stock_options
'''
## storing the modified data in my_dataset for easy export
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Task 4: Tuning the classifiers

## Importing the necessary modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn import grid_search
from sklearn.pipeline import Pipeline, FeatureUnion

## defining parameter tunes for Decsion Trees and KNN
parameters_tree = {'min_samples_split':range(2,22,2)}
parameters_knn = {'n_neighbors' : [1,2,4,6,8,10], 'metric':('euclidean','manhattan','chebyshev','minkowski')}

## defining the two classifiers
tree_classifier = DecisionTreeClassifier(random_state=12)
knn_classifier = KNeighborsClassifier()

## using GridSearchCV to get automated/ optimal parameter tunes for classifiers
## This portion is commented out because the final clf is implemented as a pipeline
##clf = grid_search.GridSearchCV(knn_classifier, parameters_knn, scoring = 'recall')
##clf = grid_search.GridSearchCV(tree_classifier, parameters_tree, scoring = 'recall')


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


## automated feature selction
## changing values of K to check how does the number of features affect performance
## This is computed separatelt to the best selected features through get_support and scores methods
number_of_features = 5
selector = SelectKBest(k = number_of_features)
selector.fit(features_train, labels_train)
features_train = selector.transform(features_train)
features_test = selector.transform(features_test)

## implementing the pipeline so that feature scaling can occur in tester.py
## 1. carrying out feature scaling (KNN algorithm)
## 2. implementing Select best features
transformed_features = FeatureUnion([('scaler', MinMaxScaler()),
('selector', selector)
])
## 3. Apply classifier to transformed data
pipeline = Pipeline([
    ('features', transformed_features),
    ('KNN', knn_classifier)])

## parameters for pipeline
parameters_pipeline = {'KNN__n_neighbors' : [1,2,4,6,8,10],
'KNN__metric':('euclidean','manhattan','chebyshev','minkowski')}

## defining the final classifier
clf = grid_search.GridSearchCV(pipeline, parameters_pipeline, scoring = 'recall')

clf.fit(features_train, labels_train)
predicted = clf.predict(features_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score

## checking performance on simply split data
print "The accuracy score of the algorithm is:", accuracy_score(labels_test, predicted)
print "The precision score of the algorithm is:", precision_score(labels_test, predicted)
print "The recall score of the algorithm is:", recall_score(labels_test, predicted)

## More robust performance metrics were gotten from the tester.py script run in the final_project ipython notebook file

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
