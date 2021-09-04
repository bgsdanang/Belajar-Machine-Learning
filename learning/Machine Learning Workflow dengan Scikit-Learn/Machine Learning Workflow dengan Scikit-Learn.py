#!/usr/bin/env python
# coding: utf-8

# # Persiapan Dataset

# ## Load Sample Dataset: Iris Dataset 

# In[2]:


from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target


# ## Splitting Dataset: Training & Testing Set
# 
# Membagi dataset menjadi dua bagian yaitu training set dan testing set
# 
# * training set digunakan untuk training model
# * testing set digunakan untuk proses melakukan evaluasi atau testing performa dari model yang kita training sebelumnya

# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = 0.4,
                                                   random_state = 1)


# # Training Model
# 
# * Pada Scikit Learn, Model machine learning dibentuk dari class yang dikenal bernama estimator
# * Setiap estimator mengimplementasikan dua method utama, yaitu `fit()` dan `predict()`
# * Method `fit()` digunakan untuk melakukan training model
# * Method `predict()` digunakan untuk melakukan estimasi/prediksi dengan memanfaatkan trained model

# In[4]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier (n_neighbors = 3) # pembentukan objek dengan parameter neighbors 3
model.fit(X_train, y_train)


# # Evaluasi Model

# In[5]:


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test) # melakukan prediksi terhadap nilai features atau X_test
acc = accuracy_score(y_test, y_pred) # membandingkan nilai target/y_test dengan nilai y_pred 
print(f'Accuracy: {acc}')


# dari hasil diatas nilai akurasi nya sebesar 0.9833 atau 98% yang termasuk kedalam nilai akurasi yang sangat baik

# # Pemanfaatan Trained Model
# 
# setelah model ini dirasa cukup baik setelah evaluasi, maka model ini dapat digunakan untuk melakukan prediksi terhadap data baru

# In[6]:


data_baru = [[5, 5, 3, 2],
            [2, 4, 3, 5]] # terdiri dari 2 arrow atau baris, setiap baris terdiri dr 4 features

preds = model.predict(data_baru) # melakukan prediksi terhadap data baru
preds


# dari hasil diatas, artinya untuk baris pertama dengan nilai features [5, 5, 3, 2] diprediksi memiliki nilai terget 1 sedangkan [2, 4, 3, 5] diprediksi memiliki nilai target 2

# In[7]:


pred_species = [iris.target_names[p] for p in preds] # memanggil target names yang dicocokan dengan hasil prediksi diatas

print(f'Hasil Prediksi: {pred_species}')


# berdasarkan output diatas, untuk baris pertama dengan nilai features [5, 5, 3, 2] diprediksi masuk klasifikasi species iris *versicolor*, sedangkan [2, 4, 3, 5] diprediksi masuk ke dalam kategori species iris *virginica*

# # Dump & Load Trained Model 

# ## Dumping Model Machine Learning menjadi file `joblib`
# 
# train model yang sudah siap akan diekspor atau dump dengan menggunakan `joblib`

# In[8]:


import joblib

# parameter pertama adlh train model yang akan di dump, parameter kedua adlh nama filenya
joblib.dump(model, 'iris_classifier_knn.joblib') 


# setelah dieksekusi maka akan terbentuk file baru dengan nama `iris_classifier_knn.joblib`

# ## Loading Model Machine Learning dari file `joblib`

# In[9]:


# nge-load `iris_classifier_knn.joblib` menjadi Machine Learning model yang siap digunakan
production_model = joblib.load('iris_classifier_knn.joblib')


# In[ ]:




