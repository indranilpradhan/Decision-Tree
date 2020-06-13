import numpy as np
import pandas as pd
from numpy.random import RandomState
from mlxtend.preprocessing import minmax_scaling
from math import log2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

class DecisionTree:
    
    def __init__(self, max_depth=7, min_element = 10):
        self.max_depth = max_depth
        self.min_element = min_element
    
    def fit(self, X_train, Y_train ):
        self.X_train = X_train
        self.Y_train = Y_train
        #self.X_validation = X_validation
        #self.Y_validation = Y_validation
        
    def fit_test(self, X_test):
        self.X_test = X_test
        
    def set_int(self):
        return 9999,9999,None
    
    def set_str(self):
        return 9999,"",None
    
    def cal_rmse(self,groups):
        l_sum = 0
        r_sum = 0
        l_temp =0
        r_temp = 0 
    #     print("len left ",len(groups[0]))
    #     print("len right ",len(groups[1]))
        if(len(groups[0]) == 0 and len(groups[1]) == 0):
            return 0
        if(len(groups[0]) != 0):
            mean_price = 0
            left_sum = 0
            for row in groups[0]:
                left_sum = left_sum + row[-1]
            mean_price = left_sum/len(groups[0])
            for row in groups[0]:
                l_temp = l_temp + ((mean_price - row[-1])**2)
            l_sum = np.sqrt(l_temp/len(groups[0]))
        if (len(groups[1]) != 0): 
            mean_price = 0
            right_sum = 0
            for row in groups[1]:
                right_sum = right_sum + row[-1]
            mean_price = right_sum/len(groups[1])
            for row in groups[1]:
                r_temp = r_temp + ((mean_price - row[-1])**2)
            r_sum = np.sqrt(r_temp/len(groups[1]))
        #l_sum = np.sqrt(np.sum((mean_price - groups[0][:,-1:])**2)/len(groups[0]))
        #r_sum = np.sqrt(np.sum((mean_price - groups[1][:,-1:])**2)/len(groups[1]))
        total_len = len(groups[0])+len(groups[1])
        result = (((len(groups[0])*l_sum)/total_len) + (((len(groups[1])*r_sum)/total_len)))
        #print("result ",l_sum)
        return result
    
    def terminate_node(self,group):
        sum_el = 0
        for row in group:
            sum_el = sum_el + row[-1]
    # print("sum/len ",sum_el/len(group))
        return sum_el/len(group)
    
    def checker(self,value):
        if (type(value) is int or type(value) is float):
            #print("srtng")
            return True
        else:
            #print("int")
            return False
        
    def test_split_num(self,index, value, train_data):
        left, right = list(), list()
        #print("index ",index)
        for row in train_data:
            #print("row index ",row[index])
            #print("value in num",value)
            if float(row[index]) <= float(value):
                left.append(row)
            else:
                right.append(row)
    #     if(index == 0):
    #         print("left ", len(left))
    #         print("right ", len(right))
        return left, right
    
    def test_split_str(self,index, value, train_data):
        left, right = list(), list()
        for row in train_data:
        # print("value in str ",value)
            if row[index] == value:
                left.append(row)
            else:
                right.append(row)
        return left, right
    
    def del_groups(self,value):
        del value
        
    def check_for_left_right(self,left,right):
        if not left or not right:
            return True
        else:
            return False
        
    def check_for_max_depth(self,max_depth):
        if max_depth < 1:
            return True
        else:
            return False
        
    def check_for_min_size(self,node,min_size):
        if (len(node) <= min_size):
            return True
        else:
            return False
        
    def split_node(self,train_data,col_size):
        int_index, int_value, int_groups = self.set_int()
        str_index, str_value, str_groups = self.set_str()
        score = 999999999999999999
        for index in range(col_size):
            for row in train_data:
                #print("row index ",row[index])
                if (self.checker(row[index])):
                #  print("if index ",index)
                    groups = self.test_split_num(index, row[index], train_data)
                    rmse = self.cal_rmse(groups)
                    if rmse < score:
#                       print("index ",index)
#                       print("rmse ",rmse)
#                       print("in loop int left groups 0", len(groups[0]))
#                        print("in loop int right groups 1", len(groups[1]))
                        int_index, int_value,score, int_groups = index, row[index], rmse, groups
                        str_groups = None
                else:
                #   print("else index ",index)
                    groups = self.test_split_str(index, row[index], train_data)
                    rmse = self.cal_rmse(groups)
                    if rmse < score:
#                       print("index ",index)
#                       print("rmse ",rmse)
#                       print("in loop str left groups 0", len(groups[0]))
#                       print("in loop str right groups 1", len(groups[1]))
                        str_index, str_value, score, str_groups = index, row[index], rmse, groups
                        int_groups = None
        if(int_groups == None):
#           print("here str left groups 0", len(str_groups[0]))
#           print("here str right groups 1", len(str_groups[1]))
            return {'index':str_index, 'value':str_value, 'groups':str_groups}
        elif(str_groups == None):
#           print("here int left groups 0",len(int_groups[0]))
#           print("here int right groups 1",len(int_groups[1]))
            return {'index':int_index, 'value':int_value, 'groups':int_groups}
#       else:
#           print("none")
#           return


    def split_tree(self,node, max_depth, min_size,col_size):
        #print("node groups ",node['groups'])
        left, right = node['groups']
        #print("left ",left)
        #print("right ",right)
        self.del_groups(node['groups'])
        if (self.check_for_left_right(left,right)) or (self.check_for_max_depth(max_depth)):
            if (self.check_for_left_right(left,right)):
                #print("bth null")
                node['left'] = node['right'] = self.terminate_node(left + right)
                return
            if (self.check_for_max_depth(max_depth)):
                #print("max depth")
                node['left'], node['right'] = self.terminate_node(left), self.terminate_node(right)
                return
        
        if (self.check_for_min_size(left,min_size)):
            #print("left terminate")
            node['left'] = self.terminate_node(left)
        else:
            #print("left len")
            node['left'] = self.split_node(left,col_size)
            self.split_tree(node['left'], max_depth-1, min_size,col_size)
        if (self.check_for_min_size(right,min_size)):
            #print("right term")
            node['right'] = self.terminate_node(right)
        else:
            #print("right len")
            node['right'] = self.split_node(right,col_size)
            self.split_tree(node['right'], max_depth-1, min_size,col_size)
            
            
    def decision_tree(self,train_data,max_depth,min_element):
        col_size = train_data.shape[1]-1
        root = self.split_node(train_data,col_size)
        #print("max_depth ",root)
        self.split_tree(root,max_depth,min_element,col_size)
        return root
    
    def make_prediction(self,root,row):
        #print("root index ",row[root['index']])
        #print("root value ",root['value'])
        if row[root['index']] <= root['value']:
            if isinstance(root['left'],dict):
                return self.make_prediction(root['left'], row)
            else:
     #          print("root left ",root['left'])
                return root['left']
        else:
            if isinstance(root['right'],dict):
                return self.make_prediction(root['right'], row)
            else:
      #         print("root right ",root['right'])
                return root['right']
          
          
    def mean_square_error(self,y_real,y_prediction):
        summation = 0
        n = len(y_real)
        for i in range (0,n):
            difference = y_real[i] - y_prediction[i]
            squared_difference = difference**2
            summation = summation + squared_difference
        MSE = summation/n
        return MSE
    
    
    def mean_absolute_error(self,y_real,y_prediction):
        summation = 0
        n = len(y_real)
        for i in range (0,n):
            difference = np.abs(y_real[i] - y_prediction[i])
            summation = summation + difference
            #print("Summation ",summation)
        MAE = summation/n
        return MAE
    
    def clean_column(self, df):
        df.drop(['Alley','PoolQC','Fence','MiscFeature'], axis =1, inplace = True)
        df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace = True)
        df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace = True)
        df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace = True)
        df['BsmtQual'].fillna(df['BsmtQual'].mode()[0], inplace = True)
        df['BsmtCond'].fillna(df['BsmtCond'].mode()[0], inplace = True)
        df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0], inplace = True)
        df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0], inplace = True)
        df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0], inplace = True)
        df['Electrical'].fillna(df['Electrical'].mode()[0], inplace = True)
        df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0], inplace = True)
        df['GarageType'].fillna(df['GarageType'].mode()[0], inplace = True)
        df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean(), inplace = True)
        df['GarageFinish'].fillna(df['GarageFinish'].mode()[0], inplace = True)
        df['GarageQual'].fillna(df['GarageQual'].mode()[0], inplace = True)
        df['GarageCond'].fillna(df['GarageCond'].mode()[0], inplace = True)
        return df
    
    def train(self,path):
        df = pd.read_csv(str(path))
        df = self.clean_column(df)
        #rng = RandomState()
        #train = df.sample(frac=0.8,random_state = rng)
        #mean_price = train["SalePrice"].mean()
        #validation = df.loc[~df.index.isin(train.index)]
        X_train, Y_train = df.iloc[:,1:], df.iloc[:,-1:]
        #X_validation, Y_validation = validation.iloc[:,1:-1], validation.iloc[:,-1:]
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        #print("x train shape", X_train.shape)
        #print("y train", Y_train.shape)
        #print("x train", X_train)
        #X_validation = np.array(X_validation)
        #Y_validation = np.array(Y_validation)
        self.fit(X_train,Y_train)
        
    def predict(self, path):
        test_df = pd.read_csv(str(path))
        test_df = self.clean_column(test_df)
        test_data = np.array(test_df.iloc[:,1:])
        #print("test shape ",test_data)
        self.fit_test(test_data)
        root = self.decision_tree(self.X_train,self.max_depth,self.min_element)
        predictions = list()
        for row in test_data:
            res = self.make_prediction(root,row)
            predictions.append(res)
        return predictions    
        
    
    
    