import pandas  as pd 
import numpy as np 
import matplotlib.pyplot as plt 


dataset_train = pd.read_csv('Electrical_load_train.csv')
dataset_train = dataset_train.drop('Unnamed: 0',axis = 1)
training_set = dataset_train[["units"]].values

from sklearn.preprocessing import MinMaxScaler 
scale = MinMaxScaler(feature_range = (0,1))
training_set_scaled = scale.fit_transform(training_set)

X_train = []
y_train = []

for x in range(12,len(training_set_scaled)):
     X_train.append(training_set_scaled[x-12:x,0])
     y_train.append(training_set_scaled[x,0])
    
X_train , y_train = np.array(X_train),np.array(y_train)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)


from tensorflow.keras.layers import Dense,Dropout,LSTM 
from tensorflow.keras.activations import relu,linear
from sklearn.model_selection import train_test_split 
from  tensorflow.keras.models import Sequential 

reg = Sequential([LSTM(units = 100,input_shape = (X_train.shape[1],1),activation = relu,
                       return_sequences = True)])

reg.add(Dropout(0.2))

reg.add(LSTM(units = 50,activation = relu,return_sequences = True))

reg.add(Dropout(0.2))

reg.add(LSTM(units = 50,activation = relu))

reg.add(Dropout(0.2))

reg.add(Dense(units = 1,activation = linear))


reg.compile(optimizer = "adam",loss = "mean_squared_error")

reg.fit(X_train,y_train,epochs = 500,batch_size = 12)

reg.save('Electrical_load_rnn_model.h5')

dataset_test = pd.read_csv('Electrical_load_test.csv')

dataset_test = dataset_test.drop('Unnamed: 0',axis = 1)

real_units = dataset_test[["units"]].values 

test = real_units

test = scale.fit_transform(test)

inputs  = np.append(training_set_scaled[-12:],test)

inputs = inputs.reshape(-1,1)


X_test = []



for x in range(12,24):
    X_test.append(inputs[x-12:x,0])


    

X_test = np.array(X_test)


   
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)


predicted_units = scale.inverse_transform(reg.predict(X_test))

plt.plot(real_units,color = 'red',label = 'Real units')

plt.plot(predicted_units,color = 'blue',label = 'Predicted units')

plt.title("ON TEST DATA")

plt.ylabel("Load (in KWhr)")

plt.legend()


plt.show()

df_predicted_test_data = pd.DataFrame({"real_units":real_units.reshape(12,),
                                       "predicted_units":predicted_units.reshape(12,)},
    index = pd.date_range('2019/01/01',periods = 12,freq = 'M'))

time = 72
test_pr = training_set_scaled[-12:].reshape(1,12,1)
predicted_Kwhr = []
for x in range(0,time):
    test_predictions = reg.predict(test_pr)
    predicted_Kwhr.append(scale.inverse_transform(test_predictions)[0])
    test_pr = np.append(test_pr[:,1:,:],test_predictions.reshape(1,1,1),axis = 1)
    
 
df = pd.DataFrame()

df_date = pd.date_range('2019/1/1',periods = time,freq = 'M')
    
for x in range(0,len(predicted_Kwhr)):
    test_kwhr = pd.DataFrame(predicted_Kwhr[x],columns = ["predicted_units"])
    df = pd.concat([df, test_kwhr],axis = 0)

df.index = df_date



all_dates = pd.date_range(start = '2013/9/1',periods = 76,
                      freq = 'M')

dataset_total = pd.concat([dataset_train,dataset_test],axis = 0)




dataset_total.index = all_dates

plt.plot(dataset_total.index,dataset_total.units,color = 'b',label = 'real_data')

plt.plot(df.index,df[["predicted_units"]],color = 'r',label = 'predicted_data')

plt.xlabel("Time")

plt.ylabel("Load (in KWhr)")

plt.legend()



plt.show()

print(df_predicted_test_data)





    




    

    

    

    

    
    
    
















    





