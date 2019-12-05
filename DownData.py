import urllib.request
import os
import pandas as pd

from sklearn import  preprocessing
data_url="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
data_file_path="data/titanic3.xls"
if not os.path.isfile(data_file_path):
    result=urllib.request.urlretrieve(data_url,data_file_path)
    print('downloaded',result)
else:
    print(data_file_path,'data file already exists')

df_data=pd.read_excel(data_file_path)

selected_cols=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
selected_df_data=df_data[selected_cols]

#定义数据预处理函数
def prepare_data(df_data):
    df=df_data.drop(['name'],axis=1)
    age_mean=df['age'].mean()
    df['age']=df['age'].fillna(age_mean)
    fare_mean=df['fare'].mean()
    df['fare']=df['fare'].fillna(fare_mean)
    df['sex']=df['sex'].map({'female':0,'male':1}).astype(int)
    df['embarked']=df['embarked'].fillna('S')
    df['embarked']=df['embarked'].map({'C':0,'Q':1,'S':2}).astype(int)

    ndarray_data=df.values
    features=ndarray_data[:,1:]
    label=ndarray_data[:,0]

    minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
    norm_features=minmax_scale.fit_transform(features)

    return norm_features,label

shuffled_df_data=selected_df_data.sample(frac=1)
x_data,y_data=prepare_data(shuffled_df_data)

train_size=int(len(x_data)*0.8)

x_train=x_data[:train_size]
y_train=y_data[:train_size]
x_test=x_data[train_size:]
y_test=y_data[train_size:]