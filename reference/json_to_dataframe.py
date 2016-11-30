
# coding: utf-8

# In[53]:

import json
import pandas as pd

data = []
file_name = 'yelp_academic_dataset_business.json'
with open(file_name) as f:
    for line in f:
        data.append(json.loads(line))


# In[54]:

print (len(data))


# In[55]:

print ("first row \n",data[0])
print ("last row \n ",data[85900])


# In[56]:

print (data[0].keys())


# In[57]:

df = pd.DataFrame(columns= data[0].keys() )


# In[58]:

print (df)


# In[59]:

for idx, val in enumerate(data[:100]):
    df = df.append(data[idx], ignore_index=True)


# In[52]:

#print (df)


# In[ ]:

#restaurants = df[df.loc['categories']]
print ( df['categories'].keys())
#categories = df['categories']
#for category_obj in categories:
   #for category in category_obj:
       #print ("Restaurants" in category_obj)
#print (restaurants)



