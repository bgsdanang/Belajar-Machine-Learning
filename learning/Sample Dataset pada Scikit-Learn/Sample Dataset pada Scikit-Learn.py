#!/usr/bin/env python
# coding: utf-8

# # Load Dataset

# In[1]:


from sklearn.datasets import load_iris

iris = load_iris()
iris


# In[2]:


iris.keys() #melihat keys apa saja yg dimiliki


# # Deskripsi dari Sample Dataset
# 
# Referensi iris flower dataset : https://en.wikipedia.org/wiki/Iris_flower_data_set

# In[3]:


print(iris.DESCR) #mengakses deskripsi sample dataset


# Judul atau nama dataset nya adalah Iris plants dataset. Number of Instances merepresentasikan jumlah baris nya, pada hasil diatas jumlah barisnya adalah 150 baris, kemudian terdapat 3 kelas dimana setiap kelasnya terdapat 50 baris. kelas nya yaitu
# - Iris-Setosa, 
# - Iris-Versicolour, 
# - Iris-Virginica. 
# 
# Lalu ada Number of Attributes atau jumlah atribut, terdapat 4 numerik atribute yaitu sepal length in cm, sepal width in cm, petal length in cm dan petal width in cm.
# 
# 
# *Summary Statistics* ==> Berisi mengenai statistika deskriptif dari data-data numeriknya, akan ada informasi nilai terkecil (MIN), nilai terbesar (MAX), Mean, standar deviasi dan class correlation (korelasi terhadap atribut terhadap class nya).
# 
# *Missing Attribute Values: None*  ==> artinya semua data nya lengkap, tidak ada data yang kosong
# 
# *Class Distribution: 33.3% for each of 3 classes* ==> distribusi nya adalah 33.3% untuk setiap class nya
# 

# # Explanatory & Response Variables (Feature & Target)

# ## Explanatory Variables (Feature)

# In[4]:


X = iris.data
X.shape
# X


# artinya dimensi data nya terdiri dari 150 baris dan 4 kolom

# In[5]:


X = iris.data
# X.shape
X


# ## Response Variable (Target)

# In[6]:


y = iris.target
y.shape
# y


# artinya dataset tersebut terdiri dari 150 baris dan 1 kolom

# In[7]:


y = iris.target
# y.shape
y


# Features dan Target saling berhubungan, misal Features dengan nilai [5.1, 3.5, 1.4, 0.2] akan berkorelasi dengan Target bernilai 0 

# # Feature and Target Names
# 
# Feature dan Target yang sudah diakses sebelumnya hanya dapat melihat sekumpulan nilai yang tersimpan dalam suatu array tetapi tidak memahami makna dari nilai-nilai tersebut. agar memahami makna tersebut maka gunakan Feature Names dan Target Names.

# In[9]:


feature_names = iris.feature_names
feature_names


# hasil diatas akan berkorelasi dengan data Explanatory Variables (Feature), urutan pertama akan merepresentasikan sepal length (cm), ke-dua adalah sepal width (cm), ketiga adalah petal length (cm), dan keempat adalah petal width (cm) 

# In[11]:


target_names = iris.target_names
target_names


# Hasilnya akan berkorelasi dengan Response Variable (Target)
# 
# index ke 0 : sentosa
# 
# index ke 1 : versicolor
# 
# index ke 2 : virginica

# # Visualisasi Data

# ## Visualisasi Sepal Length & Width

# In[14]:


import matplotlib.pyplot as plt

X = X[:, :2] # hanya menyertakan dua kolom pertama saja

x_min, x_max = X[:, 0].min() - 0.5 , X[:, 0].max() + 0.5 # akan ber-asosiasi dengan kolom sepal length
y_min, y_max = X[:, 1].min() - 0.5 , X[:, 1].max() + 0.5 # akan ber-asosiasi dengan kolom sepal width

plt.scatter(X[:, 0], X[:, 1], c=y) # membuat scatterplot
plt.xlabel('Sepal length') # memberikan label
plt.ylabel('Sepal Width') # memberikan label

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True) # memberikan grid
plt.show() # memunculkan plot


# # Training & Testing Dataset

# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = 0.3,
                                                   random_state = 1)
print(f'X train : {X_train.shape}')
print(f'X test : {X_test.shape}')
print(f'y train : {X_train.shape}')
print(f'y test : {X_test.shape}')


# # Load Iris Dataset sebagai Pandas DataFrame

# In[16]:


iris = load_iris(as_frame = True)

iris_feature_df = iris.data
iris_feature_df


# In[ ]:




