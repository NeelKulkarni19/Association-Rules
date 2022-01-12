#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 2  My movies


# In[3]:


import pandas as pd 
import numpy as np


# In[4]:


movie= pd.read_csv("F:/Dataset/my_movies.csv")


# In[5]:


movie


# In[6]:


movie.info()


# In[8]:


Movie2= movie.iloc[:,5:]


# In[9]:


Movie2


# In[12]:


from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[13]:


# 10% support 70 % confidance


# In[14]:


Frequent_Items= apriori(Movie2,min_support=0.10,use_colnames=True)


# In[15]:


Frequent_Items


# In[16]:


rules= association_rules(Frequent_Items,metric='lift',min_threshold=0.70)


# In[17]:


rules


# In[21]:


rules[rules.lift>1]


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


plt.scatter(rules.support,rules.confidence)
plt.show()


# In[ ]:


# 15 % support and 80% support


# In[24]:


Frequent_Items2=apriori(Movie2,min_support=0.15,use_colnames=True)


# In[25]:


Frequent_Items2


# In[29]:


rules2=association_rules(Frequent_Items2,metric='lift',min_threshold=0.80)


# In[30]:


rules2


# In[28]:


rules2[rules2.lift>1]


# In[32]:


plt.scatter(rules2.support,rules2.confidence)
plt.show()


# In[33]:


# 20 % support and 90% support


# In[34]:


Frequent_Items3=apriori(Movie2,min_support=0.20,use_colnames=True)


# In[35]:


Frequent_Items3


# In[36]:


rules3=association_rules(Frequent_Items3,metric='lift',min_threshold=0.90)


# In[37]:


rules3


# In[38]:


rules3[rules3.lift>1]


# In[39]:


plt.scatter(rules3.support,rules3.confidence)
plt.show()


# In[ ]:




