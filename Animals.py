import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = './PetImages'
CATEGORIES = ["Dog" , "Cat"]

# for category in CATEGORIES:
#     path = os.path.join(DATADIR , category)
#     for img in os.listdir(path):
#         Img_array = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE)
#         plt.imshow(Img_array , cmap = 'gray')
#         plt.show()
#         break
#     break

IMG_SIZE = 50

# New_array = cv2.resize(Img_array , (IMG_SIZE, IMG_SIZE))
# plt.imshow(New_array , cmap = 'gray')
# plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR , category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                Img_array=cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE)
                New_array = cv2.resize(Img_array , (IMG_SIZE, IMG_SIZE))
                training_data.append([New_array, class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X=[]
y=[]

for features , label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1 , IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in=open("X.pickle", "rb")
X=pickle.load(pickle_in)

X[1]
