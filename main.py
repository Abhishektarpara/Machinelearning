# Importing Libraries
import os
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Importing files
import Decision_tree
import Gredient_boosting
import KNN
import Naive_bayes
import Random_Forest_Classifier
import xlsxwriter

print("Creating output file ...")
path = os.getcwd()
csv_files = glob.glob(os.path.join('Dataset',"*.csv"))

i=1
KNN_Output={}
Decision_tree_output={}
Gredient_boosting_output={}
Naive_bayes_output={}
Random_Forest_Classifier_output={}
for f in csv_files:
    df = pd.read_csv(f)
    label_encoder = LabelEncoder()
    df_transformed = df.apply(label_encoder.fit_transform)
    x = df_transformed.iloc[:, :-1]
    y = df_transformed.iloc[:, -1]

    KNN_Output["Dataset" + str(i)] = KNN.knn(x,y)
    Decision_tree_output["Dataset" + str(i)] = Decision_tree.Decision_tree(x,y)
    Gredient_boosting_output["Dataset"+str(i)] = Gredient_boosting.Gredient_boosting(x,y)
    Naive_bayes_output["Dataset"+str(i)] = Naive_bayes.Naive_bayes(x,y)
    Random_Forest_Classifier_output["Dataset"+str(i)] = Random_Forest_Classifier.Random_forest_classifier(x,y)
    i += 1

classifier_name = ['Gredient boosting', 'Random forest','Naive Bayes', 'KNN', 'Decision Tree']
evaluation_method_name = ['Accuracy_score', 'Precision_score', 'Recall_score', 'F1_score', 'Jaccard_score']
dataset_name = ['Dataset1', 'Dataset2', 'Dataset3', 'Dataset4', 'Dataset5']
key = ['Mean Validation Accuracy score', 'Mean Validation Precision score', 'Mean Validation Recall score',
           'Mean Validation F1 Score','Mean validation Jaccard Score']


dname = ['dataset-Customer','dataset-Gender','dataset-House-voting','dataset-Movie-review','dataset-wine']
# Writing data into Excel file
workbook = xlsxwriter.Workbook('output.xlsx')
worksheet = workbook.add_worksheet()

# Adjust the column width.
worksheet.set_column(0, 10, 25)
# Add a bold format to use to highlight cells.
bold = workbook.add_format({'bold': 1})

rows = 0
cols = 0
indices = 0
for d in dataset_name:
    worksheet.write(rows, 0, evaluation_method_name[indices], bold)

    # writing heading in excel file
    idx = 0
    for col in range(2, 7):
        worksheet.write(rows + 2, col, classifier_name[idx], bold)
        idx += 1

    idx = 0
    for row in range(rows + 3, rows + 8):
        worksheet.write(row, 1, dname[idx], bold)
        idx += 1

    # writing content for LR
    col = cols + 2
    idx = 0
    for row in range(rows + 3, rows + 8):

        worksheet.write(row, col, KNN_Output[dataset_name[idx]][key[indices]])
        worksheet.write(row, col + 1, Decision_tree_output[dataset_name[idx]][key[indices]])
        worksheet.write(row, col + 2, Gredient_boosting_output[dataset_name[idx]][key[indices]])
        worksheet.write(row, col + 3, Naive_bayes_output[dataset_name[idx]][key[indices]])
        worksheet.write(row, col + 4, Random_Forest_Classifier_output[dataset_name[idx]][key[indices]])
        idx += 1

    rows += 10
    indices += 1

workbook.close()
print("Output file created successfully Open Output.xlsx for output")
