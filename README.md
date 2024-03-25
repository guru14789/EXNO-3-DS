## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:

STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.


# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv('/content/Encoding Data.csv')
df
```
![Screenshot 2024-03-25 091729](https://github.com/guru14789/EXNO-3-DS/assets/151705853/536e6785-3739-48bf-b567-0e480cc494a6)

# Ordinal Encoding
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-03-25 091739](https://github.com/guru14789/EXNO-3-DS/assets/151705853/237172e7-7f3e-47fe-ad23-e027a9746f36)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-03-25 091745](https://github.com/guru14789/EXNO-3-DS/assets/151705853/c5da43e0-0fec-4bd5-8d84-de85e039d7d8)

# Label Encoder
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-03-25 092057](https://github.com/guru14789/EXNO-3-DS/assets/151705853/5b3a3672-acc4-4ca1-93e0-4012950ef048)

# OneHot Encoder
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-03-25 092011](https://github.com/guru14789/EXNO-3-DS/assets/151705853/d967ce56-7bc1-4595-bca4-e1933864694e)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2024-03-25 092032](https://github.com/guru14789/EXNO-3-DS/assets/151705853/87ed636f-b57d-4968-841c-4a87eb7db300)

# Binary Encoder
```
pip install --upgrade category_encoders
```
![Screenshot 2024-03-25 092224](https://github.com/guru14789/EXNO-3-DS/assets/151705853/62152352-a03d-4df3-b830-53207c218c7d)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![Screenshot 2024-03-25 092244](https://github.com/guru14789/EXNO-3-DS/assets/151705853/99121f9e-0b60-4d0e-895a-9027872b4bef)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
![Screenshot 2024-03-25 092327](https://github.com/guru14789/EXNO-3-DS/assets/151705853/0a0871d3-503c-4635-b094-77055b9f7b33)



# Target Encoder
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2024-03-25 092400](https://github.com/guru14789/EXNO-3-DS/assets/151705853/95a1922c-a2d5-421b-8a4c-6ad29ec41213)

# Data Transformation
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/content/Data_to_Transform.csv')
df
```
![Screenshot 2024-03-25 092422](https://github.com/guru14789/EXNO-3-DS/assets/151705853/41901f94-cf76-469a-b86b-de60c4da46a8)


```
df.skew()
```
![Screenshot 2024-03-25 092459](https://github.com/guru14789/EXNO-3-DS/assets/151705853/cba217d5-450f-40cd-8df5-97cef2e2c4d9)

```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-03-25 092528](https://github.com/guru14789/EXNO-3-DS/assets/151705853/2f81024f-39c9-4c15-8e21-d352f5385cf9)

```
np.reciprocal(df["Moderate Positive Skew"])

```
![Screenshot 2024-03-25 092553](https://github.com/guru14789/EXNO-3-DS/assets/151705853/a2bf37fe-b043-4f6a-83bd-5e95d3992853)

```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2024-03-25 092625](https://github.com/guru14789/EXNO-3-DS/assets/151705853/d466c366-5d60-461a-9410-afaf781e249e)

```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2024-03-25 092647](https://github.com/guru14789/EXNO-3-DS/assets/151705853/8936e867-5c16-46aa-abe9-0bc0f2cb9d65)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-03-25 092719](https://github.com/guru14789/EXNO-3-DS/assets/151705853/41287ac0-31ba-4f8b-a41d-b9e8e779872c)

```
df.skew()
```
![Screenshot 2024-03-25 093032](https://github.com/guru14789/EXNO-3-DS/assets/151705853/380eee56-b11f-4eaa-bd21-b10c9bedf077)


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2024-03-25 092912](https://github.com/guru14789/EXNO-3-DS/assets/151705853/22022eb0-ec3f-42bd-a4ac-02a167829e9f)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-03-25 092900](https://github.com/guru14789/EXNO-3-DS/assets/151705853/c9e4d726-ad78-4e3f-bf83-7003a54bc629)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-03-25 092952](https://github.com/guru14789/EXNO-3-DS/assets/151705853/b20405dc-e3db-4667-ace6-202898c18029)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-03-25 093014](https://github.com/guru14789/EXNO-3-DS/assets/151705853/72b1b402-d60b-4675-9009-f4485a991a78)

      
# RESULT:
Finally, perform Feature Encoding and Transformation process is executed successfully.
       
