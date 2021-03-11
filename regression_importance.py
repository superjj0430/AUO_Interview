import pandas as pd
from sklearn.linear_model import LinearRegression


#讀取訓練資料
df = pd.read_csv('train_v4.csv')
y = df['Y']
x = df.iloc[:,2:]

model = LinearRegression() 

model.fit(x, y) 

importance = model.coef_ 

# for i,v in enumerate(importance): 
#     print('Feature: %0d, Score: %.5f' % (i,v))
    
a = 0
features = pd.DataFrame()
for i, v in enumerate(importance):
    if abs(v) > 0.1:
        print('Feature: %0d, Score: %.5f' % (i,v))
        b = df.iloc[:,i].to_frame()
        features = pd.concat([features,b], axis=1)
        a+=1
