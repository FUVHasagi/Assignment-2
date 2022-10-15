import numpy as np
import pandas as pd


class Classifier:
    def __init__(self, df):
        self.df = df
        self.classifier_dict = {}

    def build_classifiers(self):
        class_value_count = self.df.iloc[:, -1].value_counts()
        total_rows = len(self.df.index)
        for i in range(0, 2, 1):
            if i in class_value_count.index:
                p_class = float(class_value_count[i]) / float(total_rows)

                self.classifier_dict[i] = {}
                self.classifier_dict[i]["class="] = p_class
                class_filtered = self.df.loc[self.df.iloc[:, -1] == i]
                total_row_class = len(class_filtered.index)
                for column in self.df.columns[:-1]:
                    attr_value_count = class_filtered[column].value_counts()
                    self.classifier_dict[i][column] = {}
                    for j in range(0, 2, 1):
                        if j in attr_value_count.index:
                            P_attr = float(attr_value_count[j]) / float(total_row_class)
                            self.classifier_dict[i][column][j] = P_attr
                        else:
                            self.classifier_dict[i][column][j] = 0
            else:
                self.classifier_dict[i] = {}
                self.classifier_dict[i]["class="] = 0
                for column in self.df.columns[:-1]:
                    self.classifier_dict[i][column] = {}
                    for j in range(0, 2, 1):
                        self.classifier_dict[i][column][j] = 0

    def print_str(self):
        obj_str = ""

        for class_value in range(0, 2, 1):
            obj_str += "P(class="
            obj_str += str(class_value)
            obj_str += ")=" + "{:.2f}".format(self.classifier_dict[class_value]['class='])
            obj_str += " "
            for column in self.classifier_dict[class_value]:
                if column != 'class=':
                    for attr_value in range(0, 2, 1):
                        obj_str += "P("
                        obj_str += str(column)
                        obj_str += "="
                        obj_str += str(attr_value)
                        obj_str += "|"
                        obj_str += str(class_value)
                        obj_str += ")="
                        obj_str += "{:.2f}".format(self.classifier_dict[class_value][column][attr_value])
                        obj_str += " "
            obj_str += "\n"
        print(obj_str)
        return obj_str

    def return_value(self, df, index):
        all_P_value = np.array([0., 0.])
        for class_value in range(0, 2, 1):
            P_class = float(self.classifier_dict[class_value]['class='])
            for column in self.classifier_dict[class_value]:
                if column != 'class=':
                    instance_attr_value = df.loc[index, column]
                    P_class *= \
                        float(self.classifier_dict[class_value][column][instance_attr_value])
            all_P_value[class_value] = P_class
        return np.argmax(all_P_value)

    def train_accuracy(self):
        n = len(self.df.index)
        count_right = 0
        for i in range(0, n, 1):
            value_classifed = self.return_value(self.df, i)
            if value_classifed == self.df.iloc[i, -1]:
                count_right += 1
        ratio = float(count_right) / float(n)
        percentage = "{:4.2%}".format(ratio)
        print("Accuracy on testing set ({} instances):{}".
              format(n, percentage))
        return percentage

    def test_accuracy(self, test_df):
        n = len(test_df.index)
        count_right = 0
        for i in range(0, n, 1):
            value_classifed = self.return_value(test_df, i)
            if value_classifed == test_df.iloc[i, -1]:
                count_right += 1
        ratio = float(count_right) / float(n)
        percentage = "{:4.2%}".format(ratio)
        print("Accuracy on testing set ({} instances):{}".
              format(n, percentage))
        return percentage
