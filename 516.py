
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("C:/Users/User/Desktop/diabetes.csv")
data.head()


# In[3]:


sns.countplot(x="Outcome", data= data)
plt.show()


# In[52]:


sns.countplot(x="Age", data=data)
plt.show()


# In[4]:


X = data.drop("Outcome", axis = 1)
y = data["Outcome"]


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, random_state=20, test_size=0.2)


# In[6]:


correlations = data.corr()
correlations['Outcome'].sort_values(ascending=False)


# In[47]:


corr = data.corr()
print(corr)


# In[31]:


sns.heatmap(data.corr())
plt.show()


# In[53]:


coeff = list(diabetesCheck.coef_[0])
labels = list(data.columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')


# In[41]:


corr = data.corr()
print (corr['Outcome'].sort_values(ascending=False)[:5], '\n')
print (corr['Outcome'].sort_values(ascending=False)[-5:])


# In[34]:


def visualise(data):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:,1].values, data.iloc[:,5].values)
    ax.set_title('Highly Correlated Features')
    ax.set_xlabel('Glucose')
    ax.set_ylabel('BMI')

visualise(data)


# In[42]:


def visualise(data):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:,1].values, data.iloc[:,7].values)
    ax.set_title('Highly Correlated Features')
    ax.set_xlabel('Glucose')
    ax.set_ylabel('Age')

visualise(data)


# In[43]:


def visualise(data):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:,1].values, data.iloc[:,4].values)
    ax.set_title('Highly Correlated Features')
    ax.set_xlabel('Glucose')
    ax.set_ylabel('Insulin')

visualise(data)


# In[44]:


def visualise(data):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:,1].values, data.iloc[:,3].values)
    ax.set_title('Highly Correlated Features')
    ax.set_xlabel('Glucose')
    ax.set_ylabel('SkinThickness')

visualise(data)


# In[45]:


def visualise(data):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:,1].values, data.iloc[:,2].values)
    ax.set_title('Highly Correlated Features')
    ax.set_xlabel('Glucose')
    ax.set_ylabel('Blood Presure')

visualise(data)


# In[46]:


def visualise(data):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:,1].values, data.iloc[:,0].values)
    ax.set_title('Highly Correlated Features')
    ax.set_xlabel('Glucose')
    ax.set_ylabel('Pregnancies ')

visualise(data)


# In[39]:


def visualise(data):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:,1].values, data.iloc[:,8].values)
    ax.set_title('Highly Correlated Features')
    ax.set_xlabel('Glucose')
    ax.set_ylabel('Outcome')

visualise(data)


# In[16]:


Glucose= data.iloc[:,1:2]


# In[23]:


Glucose.describe()


# In[25]:


print ("Skew is:", data.Glucose.skew())
plt.hist(data.Glucose, color='blue')
plt.show()


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(data.Glucose,data.Outcome).plot(kind='bar')
plt.title('Glucose levels of diabetes and non diabetes')
plt.xlabel('Glucose Range')
plt.ylabel('Glucose levels')
#plt.savefig('purchase_fre_job')


# In[29]:


table=pd.crosstab(data.Glucose,data.Outcome)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Glucose levels of diabetes and non diabetes')
plt.xlabel('Glucose Range')
plt.ylabel('Glucose levels')
#plt.savefig('mariral_vs_pur_stack')


# In[21]:


plt.plot(Glucose)


# In[22]:


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

# store the predictions
y_pred=logreg.predict(X_validation)


# In[35]:


from sklearn import metrics
cf_matrix = metrics.confusion_matrix(y_validation, y_pred)
cf_matrix


# In[36]:


sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# In[37]:


print("Accuracy:",metrics.accuracy_score(y_validation, y_pred))
print("Precision:",metrics.precision_score(y_validation, y_pred))
print("Recall:",metrics.recall_score(y_validation, y_pred))


# In[38]:


y_pred_proba = logreg.predict_proba(X_validation)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_validation,  y_pred_proba)
auc = metrics.roc_auc_score(y_validation, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# In[55]:


coeff = list(logreg.coef_[0])
labels = list(X_train.columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')

