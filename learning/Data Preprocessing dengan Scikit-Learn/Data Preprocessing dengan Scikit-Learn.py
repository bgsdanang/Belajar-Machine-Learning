#!/usr/bin/env python
# coding: utf-8

# # Sample Data

# In[4]:


import numpy as np
from sklearn import preprocessing

sample_data = np.array([[2.1, -1.9, 5.5],
                       [-1.5, 2.4, 3.5],
                       [0.5, -7.9, 5.6],
                       [5.9, 2.3, -5.8]])

sample_data


# Dataset diatas bisa dipandang sebagai dataset features.

# In[5]:


sample_data.shape # menampilkan dimensi dari dataset diatas


# artinya jumlah baris nya yaitu ada 4 dan jumlah kolom nya ada 3

# # Teknik Binarisation
# 
# Teknik ini bertujuan untuk menghasilkan sebuah suatu data yang terdiri dari dua nilai numerik saja, yaitu 0 dan 1

# In[6]:


sample_data # dataset yang telah terbentuk


# Dataset tersebut terdiri dari sekumpulan nilai plotting poin yang cukup beragam. Misal kita ingin mengkonversikan setiap nilai numerik yang lebih besar dari 0.5 menjadi 1 dan sisanya dikonversi menjadi 0. Code nya bisa dilihat dibawah ini

# In[7]:


preprocessor = preprocessing.Binarizer(threshold = 0.5) # parameter threshold diisi 0.5
binarised_data = preprocessor.transform(sample_data)
binarised_data


# bisa dilihat hasil dari proses binarisation adalah suatu dataset yang hanya terdiri dari dua nilai, yaitu 0 dan 1. Dimana batasan nya ditentukan dari nilai `threshold`. Jika setiap nilai <= 0.5 akan dikonversikan menjadi nilai 0, sedangkan nilai-nilai yang > 0.5 maka dikonversikan menjadi 1

# # Teknik Scalling
# 
# Tujuan dari teknik *scalling* ini untuk menghasilkan suatu data numerik yang dalam rentang skala tertentu

# In[8]:


sample_data # dataset yang dimiliki


# Dataset tersebut terdiri dari sekumpulan nilai plotting poin dengan rentang mulai dari nilai -7.9 sampai nilai 5.9. Misal kita ingin mengkonverskan sekumpulan nilai dataset diatas dalam rentang nilai 0 sampai 1. Code nya bisa dilihat dibawah ini

# In[10]:


preprocessor = preprocessing.MinMaxScaler(feature_range=(0, 1)) # parameter feature_range diisi 0 sampai 1
preprocessor.fit(sample_data) 

scaled_data = preprocessor.transform(sample_data) # transformasi data
scaled_data


# Dari hasil diatas, nilai terkecil nya 0 dan nilai terbesar 1. Dan output diatas hasil dari transformasi dari variabel `sample_data`. Jadi bisa disimpulkan proses `MinMaxScaler` ini digunakan untuk mengubah skala nilai terkecil dan terbesar dari dataset yang kita punya ke skala tertentu

# Untuk code yang lebih singkat bisa lihat dibawah ini.

# In[11]:


scaled_data = preprocessor.fit_transform(sample_data)
scaled_data


# # L1 Normalisation: Least Absolute Deviations

# Tujuan utama dari teknik ini adalah untuk melakukan normalisasi terhadap data numerik yang kita miliki.

# In[13]:


sample_data


# In[14]:


# teknik ini butuh 2 parameter, yg pertama data kita dan yg kedua parameter norm
# l1 artinya mengarah ke teknik normalisasi Least Absolute Deviations
l1_normalised_data = preprocessing.normalize(sample_data, norm='l1') 
l1_normalised_data


# # L2 Normalisation: Least Square

# In[15]:


sample_data


# In[16]:


# teknik ini butuh 2 parameter, yg pertama data kita dan yg kedua parameter norm
# l2 artinya mengarah ke teknik normalisasi Least Square
l2_normalised_data = preprocessing.normalize(sample_data, norm='l2') 
l2_normalised_data


# In[ ]:




