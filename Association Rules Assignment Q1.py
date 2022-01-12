#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Question 1 Book 


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


book= pd.read_csv('F:/Dataset/book.csv')


# In[3]:


book


# In[4]:


book.info()


# In[5]:


from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[26]:


# 10 % support and 70% confidance


# In[6]:


FrequentItemsets= apriori(book,min_support=0.1,use_colnames=True) 


# In[7]:


FrequentItemsets


# In[11]:


rules= association_rules(FrequentItemsets,metric='lift',min_threshold=0.7)


# In[12]:


rules


# In[14]:


rules.sort_values('lift',ascending=False)


# In[15]:


rules[rules.lift>1]


# In[16]:


import matplotlib.pyplot as plt


# In[19]:


plt.scatter(rules.support,rules.confidence)
plt.show()


# In[ ]:


# 20 % support and 75 % confidance


# In[21]:


FrequentItemsets2 = apriori(book,min_support=0.20,use_colnames=True)


# In[22]:


FrequentItemsets2


# In[27]:


rules2=association_rules(FrequentItemsets2,metric='lift', min_threshold=0.75)


# In[28]:


rules2


# In[32]:


plt.scatter(rules2.support,rules2.confidence)
plt.show()


# In[ ]:


# 5% support and 80% confidance


# In[35]:


FrequentItemsets3= apriori(book,min_support=0.05, use_colnames=True)


# In[36]:


FrequentItemsets3


# In[37]:


rules3= association_rules(FrequentItemsets3,metric='lift',min_threshold=0.80)


# In[38]:


rules3


# In[39]:


rules3[rules3.lift>1]


# In[40]:


plt.scatter(rules3.support,rules3.confidence)
plt.show


# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:





# In[43]:





# In[44]:





# In[ ]:





# In[ ]:





# In[47]:





# In[ ]:





# In[69]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




