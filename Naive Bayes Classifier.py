from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import numpy as np

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
        'Rainy','Sunny','Overcast','Overcast','Rainy']

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


le = preprocessing.LabelEncoder()
wheather_encoded = le.fit_transform(weather)
temp_encoded = le.fit_transform(temp)

labels = le.fit_transform(play)

print(wheather_encoded)
print(temp_encoded)
print(labels)

#combine two features

features = zip(wheather_encoded , temp_encoded)

features_list = list(features)



model = GaussianNB()
model.fit(features_list , labels)

#prediction

z = np.array([[0,2] , [2,2]])
z= z.reshape(len(z),-1)

prediction = model.predict(z)
print(prediction)

#1 for play yes
#2 for play no
