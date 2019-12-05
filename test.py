import tensorflow as tf
import pandas as pd
import os
from model import model,checkpoint_path
from DownData import  x_test,y_test,selected_cols,selected_df_data,prepare_data

evaluate_result=model.evaluate(x=x_test,y=y_test)
print(evaluate_result)


# checkpoint_dir=os.path.dirname(checkpoint_path)
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model.load_weights(latest)

Jake_info=[0,'Jake',3,'male',23,1,0,5.0000,'S']
Rose_info=[1,'Rose',1,'female',20,1,0,100.0000,'S']
new_passenger_pd=pd.DataFrame([Jake_info,Rose_info],columns=selected_cols)
all_passenger_pd=selected_df_data.append(new_passenger_pd)
x_features,y_label=prepare_data(all_passenger_pd)
surv_probability=model.predict(x_features)
all_passenger_pd.insert(len(all_passenger_pd.columns),'surv_probability',surv_probability)
print(all_passenger_pd[-5:])