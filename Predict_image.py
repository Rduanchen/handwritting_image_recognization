# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 23:12:28 2022

@author: cheny
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import glob,cv2

def show_images_labels_predictions(images,labels,predictions,start_id,num=30):
    plt.gcf().set_size_inches(12, 20)
    if num>25: num=25 
    for i in range(0, num):
        print(predictions[start_id])
        ax=plt.subplot(5,5, i+1)
        ax.imshow(images[start_id], cmap='binary')  #顯示黑白圖片
        if( len(predictions) > 0 ) :  #有傳入預測資料
            title = 'ai = ' + str(predictions[start_id])
            # 預測正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if predictions[start_id]==labels[start_id] else ' (x)') 
            title += '\nlabel = ' + str(labels[start_id])
        else :  #沒有傳入預測資料
            title = 'label = ' + str(labels[start_id])
        ax.set_title(title,fontsize=12)  #X,Y軸不顯示刻度
        ax.set_xticks([]);ax.set_yticks([])        
        start_id+=1 
    plt.show()
files = glob.glob("A_test\*.jpg")  #建立測試資料
print(files)
test_feature=[]
print(test_feature)
test_label=[]
print(test_label)
for file in files:
    print(file)
    img=cv2.imread(file)
    img = cv2.resize(img, (28, 28))
    cv2.imwrite(file, img)
    img=cv2.imread(file)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #灰階    
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) #轉為反相黑白 
    test_feature.append(img)
    label=file[7:8]  #"imagedata\1.jpg"第10個字元1為label
    test_label.append(int(label))
   
test_feature=np.array(test_feature) # 串列轉為矩陣 
test_label=np.array(test_label)     # 串列轉為矩陣
test_feature_vector = test_feature.reshape(len( test_feature), 784).astype('float32')
test_feature_normalize = test_feature_vector/255
model = load_model('Moduel_for_image.h5')

predictions=model.predict(test_feature_normalize) 
predictions=np.argmax(predictions,axis=1)
show_images_labels_predictions(test_feature,test_label,predictions,0)