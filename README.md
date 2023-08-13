Data scientist salaries analysis
===
*Author: Francisco Javier Sánchez Panduro*\
*Supervised by: Professor Doctor Brenda García Maya*\
*Monterrey Institute of Tecnology and Higher Studies*\
*13 of August 2023*

## Introduction
Using the linear regression model, we aim to predict salaries in dollars for data scientists. Using the features experience level, salary, type of job and remote radio.


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
```

Data preparation
---

(Bhatia, n.d.)

The data includes 11 columns, here explained
| Column | Description |
|---|---|
|work_year	| The year the salary was paid.|
|experience_level|	The experience level in the job during the year with the following possible values: EN Entry-level / Junior MI Mid-level / Intermediate SE Senior-level / Expert EX Executive-level / Director|
|employment_type|	The type of employement for the role: PT Part-time FT Full-time CT Contract FL Freelance|
|job_title	|The role worked in during the year.|
|salary	|The total gross salary amount paid.|
|salary_currency|	The currency of the salary paid as an ISO 4217 currency code.|
|salary_in_usd|	The salary in USD (FX rate divided by avg. USD rate for the respective year via fxdata.foorilla.com).|
|employee_residence|	Employee's primary country of residence in during the work year as an ISO 3166 country code.|
|remote_ratio|	The overall amount of work done remotely, possible values are as follows: 0 No remote work (less than 20%) 50 Partially remote 100 Fully remote (more than 80%)|
|company_location|	The country of the employer's main office or contracting branch as an ISO 3166 country code.|
|company_size|	The average number of people that worked for the company during the year: S less than 50 employees (small) M 50 to 250 employees (medium) L more than 250 employees (large)|


```python
df = pd.read_csv('ds_salaries.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>work_year</th>
      <th>experience_level</th>
      <th>employment_type</th>
      <th>job_title</th>
      <th>salary</th>
      <th>salary_currency</th>
      <th>salary_in_usd</th>
      <th>employee_residence</th>
      <th>remote_ratio</th>
      <th>company_location</th>
      <th>company_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2020</td>
      <td>MI</td>
      <td>FT</td>
      <td>Data Scientist</td>
      <td>70000</td>
      <td>EUR</td>
      <td>79833</td>
      <td>DE</td>
      <td>0</td>
      <td>DE</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2020</td>
      <td>SE</td>
      <td>FT</td>
      <td>Machine Learning Scientist</td>
      <td>260000</td>
      <td>USD</td>
      <td>260000</td>
      <td>JP</td>
      <td>0</td>
      <td>JP</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2020</td>
      <td>SE</td>
      <td>FT</td>
      <td>Big Data Engineer</td>
      <td>85000</td>
      <td>GBP</td>
      <td>109024</td>
      <td>GB</td>
      <td>50</td>
      <td>GB</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2020</td>
      <td>MI</td>
      <td>FT</td>
      <td>Product Data Analyst</td>
      <td>20000</td>
      <td>USD</td>
      <td>20000</td>
      <td>HN</td>
      <td>0</td>
      <td>HN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2020</td>
      <td>SE</td>
      <td>FT</td>
      <td>Machine Learning Engineer</td>
      <td>150000</td>
      <td>USD</td>
      <td>150000</td>
      <td>US</td>
      <td>50</td>
      <td>US</td>
      <td>L</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.shape)
```

    (607, 12)



```python
df.isnull().sum()
```




    Unnamed: 0            0
    work_year             0
    experience_level      0
    employment_type       0
    job_title             0
    salary                0
    salary_currency       0
    salary_in_usd         0
    employee_residence    0
    remote_ratio          0
    company_location      0
    company_size          0
    dtype: int64




```python
# Create dataframe with only relevant data
df = pd.DataFrame({'experience_level': df['experience_level'], 'employment_type' : df['employment_type'], 'salary_in_usd' : df['salary_in_usd'], 'salary' : df['salary'], 'remote_ratio' : df['remote_ratio']})
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>experience_level</th>
      <th>employment_type</th>
      <th>salary_in_usd</th>
      <th>salary</th>
      <th>remote_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MI</td>
      <td>FT</td>
      <td>79833</td>
      <td>70000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SE</td>
      <td>FT</td>
      <td>260000</td>
      <td>260000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SE</td>
      <td>FT</td>
      <td>109024</td>
      <td>85000</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MI</td>
      <td>FT</td>
      <td>20000</td>
      <td>20000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SE</td>
      <td>FT</td>
      <td>150000</td>
      <td>150000</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df['experience_level'].unique())
```

    ['MI' 'SE' 'EN' 'EX']



```python
print(df['employment_type'].unique())
```

    ['FT' 'CT' 'PT' 'FL']



```python
# Create dummy variables to represent categorical data in numerical form
dummies_experience_level = pd.get_dummies(df['experience_level'], prefix='experience_level', dtype = 'uint8')
dummies_experience_level.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>experience_level_EN</th>
      <th>experience_level_EX</th>
      <th>experience_level_MI</th>
      <th>experience_level_SE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummies_employment_type = pd.get_dummies(df['employment_type'], prefix='employment_type', dtype = 'uint8')
dummies_employment_type.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>employment_type_CT</th>
      <th>employment_type_FL</th>
      <th>employment_type_FT</th>
      <th>employment_type_PT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.concat([df, dummies_employment_type, dummies_experience_level], axis=1)
df.drop('experience_level', axis = 1, inplace=True)
df.drop('employment_type', axis = 1, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary_in_usd</th>
      <th>salary</th>
      <th>remote_ratio</th>
      <th>employment_type_CT</th>
      <th>employment_type_FL</th>
      <th>employment_type_FT</th>
      <th>employment_type_PT</th>
      <th>experience_level_EN</th>
      <th>experience_level_EX</th>
      <th>experience_level_MI</th>
      <th>experience_level_SE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79833</td>
      <td>70000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>260000</td>
      <td>260000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>109024</td>
      <td>85000</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20000</td>
      <td>20000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>150000</td>
      <td>150000</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Correlation

The following correlation matrix displays the Pearson correlation coefficients between multiple variables in the dataset. The Pearson correlation coefficient $r$ quantifies the strength and direction of the linear relationship between two variables.

A positive $r$ value indicates a positive correlation; the closer the value is to 1, the stronger the positive correlation. A negative value indicates the opposite, with the value closer to -1 indicating a stronger negative correlation.

To calculate $r$ between two variables $X$ and $Y$, the formula is:
$$
r = \frac{\sum{(X_i - \bar{X})(Y_i - \bar{Y})}}{\sqrt{\sum{(X_i - \bar{X})^2} \cdot \sum{(Y_i - \bar{Y})^2}}}
$$
Where:
- $X_i$ and $Y_i$ are individual data points for variables $X$ and $Y$.
- $ \bar{X} $ and $ \bar{Y} $ are the means of variables $X$ and $Y$.


```python
correlation_matrix = df.corr()
display(correlation_matrix)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary_in_usd</th>
      <th>salary</th>
      <th>remote_ratio</th>
      <th>employment_type_CT</th>
      <th>employment_type_FL</th>
      <th>employment_type_FT</th>
      <th>employment_type_PT</th>
      <th>experience_level_EN</th>
      <th>experience_level_EX</th>
      <th>experience_level_MI</th>
      <th>experience_level_SE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>salary_in_usd</th>
      <td>1.000000</td>
      <td>-0.083906</td>
      <td>0.132122</td>
      <td>0.092907</td>
      <td>-0.073863</td>
      <td>0.091819</td>
      <td>-0.144627</td>
      <td>-0.294196</td>
      <td>0.259866</td>
      <td>-0.252024</td>
      <td>0.343513</td>
    </tr>
    <tr>
      <th>salary</th>
      <td>-0.083906</td>
      <td>1.000000</td>
      <td>-0.014608</td>
      <td>-0.008268</td>
      <td>-0.014568</td>
      <td>0.025685</td>
      <td>-0.020006</td>
      <td>-0.015845</td>
      <td>0.014130</td>
      <td>0.074626</td>
      <td>-0.065995</td>
    </tr>
    <tr>
      <th>remote_ratio</th>
      <td>0.132122</td>
      <td>-0.014608</td>
      <td>1.000000</td>
      <td>0.065149</td>
      <td>-0.016865</td>
      <td>-0.023834</td>
      <td>-0.002935</td>
      <td>-0.010490</td>
      <td>0.041208</td>
      <td>-0.127850</td>
      <td>0.113071</td>
    </tr>
    <tr>
      <th>employment_type_CT</th>
      <td>0.092907</td>
      <td>-0.008268</td>
      <td>0.065149</td>
      <td>1.000000</td>
      <td>-0.007423</td>
      <td>-0.506989</td>
      <td>-0.011795</td>
      <td>0.066013</td>
      <td>0.070739</td>
      <td>-0.028817</td>
      <td>-0.047768</td>
    </tr>
    <tr>
      <th>employment_type_FL</th>
      <td>-0.073863</td>
      <td>-0.014568</td>
      <td>-0.016865</td>
      <td>-0.007423</td>
      <td>1.000000</td>
      <td>-0.453089</td>
      <td>-0.010541</td>
      <td>-0.033537</td>
      <td>-0.017229</td>
      <td>0.068108</td>
      <td>-0.034520</td>
    </tr>
    <tr>
      <th>employment_type_FT</th>
      <td>0.091819</td>
      <td>0.025685</td>
      <td>-0.023834</td>
      <td>-0.506989</td>
      <td>-0.453089</td>
      <td>1.000000</td>
      <td>-0.719987</td>
      <td>-0.167828</td>
      <td>-0.008698</td>
      <td>-0.006597</td>
      <td>0.128381</td>
    </tr>
    <tr>
      <th>employment_type_PT</th>
      <td>-0.144627</td>
      <td>-0.020006</td>
      <td>-0.002935</td>
      <td>-0.011795</td>
      <td>-0.010541</td>
      <td>-0.719987</td>
      <td>1.000000</td>
      <td>0.204028</td>
      <td>-0.027379</td>
      <td>-0.013805</td>
      <td>-0.119762</td>
    </tr>
    <tr>
      <th>experience_level_EN</th>
      <td>-0.294196</td>
      <td>-0.015845</td>
      <td>-0.010490</td>
      <td>0.066013</td>
      <td>-0.033537</td>
      <td>-0.167828</td>
      <td>0.204028</td>
      <td>1.000000</td>
      <td>-0.087108</td>
      <td>-0.302761</td>
      <td>-0.381033</td>
    </tr>
    <tr>
      <th>experience_level_EX</th>
      <td>0.259866</td>
      <td>0.014130</td>
      <td>0.041208</td>
      <td>0.070739</td>
      <td>-0.017229</td>
      <td>-0.008698</td>
      <td>-0.027379</td>
      <td>-0.087108</td>
      <td>1.000000</td>
      <td>-0.155539</td>
      <td>-0.195751</td>
    </tr>
    <tr>
      <th>experience_level_MI</th>
      <td>-0.252024</td>
      <td>0.074626</td>
      <td>-0.127850</td>
      <td>-0.028817</td>
      <td>0.068108</td>
      <td>-0.006597</td>
      <td>-0.013805</td>
      <td>-0.302761</td>
      <td>-0.155539</td>
      <td>1.000000</td>
      <td>-0.680373</td>
    </tr>
    <tr>
      <th>experience_level_SE</th>
      <td>0.343513</td>
      <td>-0.065995</td>
      <td>0.113071</td>
      <td>-0.047768</td>
      <td>-0.034520</td>
      <td>0.128381</td>
      <td>-0.119762</td>
      <td>-0.381033</td>
      <td>-0.195751</td>
      <td>-0.680373</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Find the high positive correlation values
high_positive_correlation = np.where((correlation_matrix > 0.95) & (correlation_matrix < 1))
# Print the index of values found
for i in high_positive_correlation:
    print(i)
```

    []
    []



```python
# Find the high negative correlation values
high_negative_correlation = np.where((correlation_matrix < -0.95) & (correlation_matrix > -1))
# Print the index of values found
for i in high_negative_correlation:
    print(i)
```

    []
    []


There are no high positive or negative correlation values, which implies a low linear association and suggests that our model may have weak predictive power. There is also the possibility of other types of non-linear relationships. We will further explore linear regression in this document.

---
## Citations
Bhatia, R. (n.d.). Data Science Job Salaries, V1.0. Retrieved August 11, 2023 from <a href="https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries">https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries</a>.

---
Francisco Javier Sánchez Panduro A01639832


```python

```
