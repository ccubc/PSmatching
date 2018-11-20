
# Does Owning an Energy Efficient Vehicle Lead to Longer Driving Distance

This project aims to explore the question "Does owning an _energy efficient vehicle_ lead to longer driving distance". Using data from [National Household Travel Survey(2017)](https://nhts.ornl.gov/), I explore how households' driving pattern is correlated with owning an _energy efficient vehicle_, which includes hybrid electric vehicles(HEV), plug-in hybrid electric vehicles(PHEV), electric vehicles(EV), and other alternative fuel vehicles. 

The question could be of interest to policy makers who provide financial incentives for purchasing _energy efficient vehicles_. Policy makers promote _energy efficient vehicles_ with a hope to reduce the environmental impact of driving. However, if there exists the notorious _rebound effect_, which means "owning a green vehicle leads to more driving", the environmental benefit of driving a green vehicle would be discounted. Therefore it would be benificial to the policy maker to detect and quantify such a _rebound effect_.

A main difficulty of quantifying rebound effect is "selection bias": households who anticipate to drive longer mileage have greater incentive to purchase energy efficient vehicles due to fuel cost saving. Not addressing this issue will result in over-estimate in the rebound effect. To alleviate such concern, I use propensity score matching method to first pair up households with similar characteristics and are equally likely to purchase energy efficient vehicles, then compare the difference of their driving distances. Since the paired households are believed to be equally likely to purchase energy efficient vehicles, the purchase decision becomes quasi-random. Therefore, we overcome the selection bias problem.  

The dataset contains information regarding to: 
* households' size, income, state, urban/rural area, number of adults, number of vehicles, etc.
* vehicles' fuel type, size, annual mileage, etc.

The following code will first import and clean the dataset, and then use propensity score matching method to calculate how much extra mileage are caused by owning a green vehicle.




```python

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import warnings
warnings.filterwarnings('ignore')
from pymatch.Matcher import Matcher
import statsmodels.api as sm
import seaborn as sns

```

## Import and clean dataset


```python
data = pd.read_csv('/Users/chengchen/Dropbox/NHTS_2018/data/NHTS2017/vehpub.csv')
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HOUSEID</th>
      <th>VEHID</th>
      <th>VEHYEAR</th>
      <th>VEHAGE</th>
      <th>MAKE</th>
      <th>MODEL</th>
      <th>FUELTYPE</th>
      <th>VEHTYPE</th>
      <th>WHOMAIN</th>
      <th>OD_READ</th>
      <th>...</th>
      <th>HH_CBSA</th>
      <th>HBHTNRNT</th>
      <th>HBPPOPDN</th>
      <th>HBRESDN</th>
      <th>HTEEMPDN</th>
      <th>HTHTNRNT</th>
      <th>HTPPOPDN</th>
      <th>HTRESDN</th>
      <th>SMPLSRCE</th>
      <th>WTHHFIN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30000007</td>
      <td>1</td>
      <td>2007</td>
      <td>10</td>
      <td>49</td>
      <td>49032</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>69000</td>
      <td>...</td>
      <td>XXXXX</td>
      <td>20</td>
      <td>1500</td>
      <td>750</td>
      <td>750</td>
      <td>50</td>
      <td>1500</td>
      <td>750</td>
      <td>2</td>
      <td>187.31432</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30000007</td>
      <td>2</td>
      <td>2004</td>
      <td>13</td>
      <td>49</td>
      <td>49442</td>
      <td>1</td>
      <td>2</td>
      <td>-8</td>
      <td>164000</td>
      <td>...</td>
      <td>XXXXX</td>
      <td>20</td>
      <td>1500</td>
      <td>750</td>
      <td>750</td>
      <td>50</td>
      <td>1500</td>
      <td>750</td>
      <td>2</td>
      <td>187.31432</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30000007</td>
      <td>3</td>
      <td>1998</td>
      <td>19</td>
      <td>19</td>
      <td>19014</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>120000</td>
      <td>...</td>
      <td>XXXXX</td>
      <td>20</td>
      <td>1500</td>
      <td>750</td>
      <td>750</td>
      <td>50</td>
      <td>1500</td>
      <td>750</td>
      <td>2</td>
      <td>187.31432</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30000007</td>
      <td>4</td>
      <td>1997</td>
      <td>20</td>
      <td>19</td>
      <td>19021</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>-88</td>
      <td>...</td>
      <td>XXXXX</td>
      <td>20</td>
      <td>1500</td>
      <td>750</td>
      <td>750</td>
      <td>50</td>
      <td>1500</td>
      <td>750</td>
      <td>2</td>
      <td>187.31432</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30000007</td>
      <td>5</td>
      <td>1993</td>
      <td>24</td>
      <td>20</td>
      <td>20481</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>300000</td>
      <td>...</td>
      <td>XXXXX</td>
      <td>20</td>
      <td>1500</td>
      <td>750</td>
      <td>750</td>
      <td>50</td>
      <td>1500</td>
      <td>750</td>
      <td>2</td>
      <td>187.31432</td>
    </tr>
  </tbody>
</table>
<p>5 rows Ã 49 columns</p>
</div>




```python
data = data[data.FUELTYPE<4] 
data = data[data.FUELTYPE>0] 
data = data[data.ANNMILES>0] 
data = data[data.HHFAMINC>0] 
data = data[data.HOMEOWN>0] 
data = data[data.VEHAGE>0] 
data = data[data.VEHTYPE>0]# drop the observations where some information is missing
income_dic = {1:5000, 2:12500, 3:20000, 4:30000, 5:42500, 6:62500, 7:87500, 8:112500, 9: 137500, 10: 175000, 11:225000}
data['income'] = data['HHFAMINC'].map(income_dic)
# map the mean income amount of the income category in the survey
home_dic = {1:1, 2:0, 97:0}
# yes: owning home  no: not owning home
data['homeown'] = data['HOMEOWN'].map(home_dic)
urban_dic = {1:'urban_area',2:'urban_cluster',3:'near_urban',4:'not_urban'}
data['urban'] = data['URBAN'].map(urban_dic)
vehtype_dic = {1: 'car',2: 'van',3: 'SUV',4: 'pickup',5: 'truck',6: 'RV',7: 'motorcycle',97: 'else'}
data['vehtype'] = data['VEHTYPE'].map(vehtype_dic)
fueltype_dic = {1: 'gas', 2: 'diesel', 3: 'hybrid/electric/alternative'}
data['fueltype'] = data['FUELTYPE'].map(fueltype_dic)

```

## Summary Statistics


```python
# Frequency of FuelType
data.groupby('fueltype')['fueltype'].count()
```




    fueltype
    diesel                           5362
    gas                            170384
    hybrid/electric/alternative      4966
    Name: fueltype, dtype: int64




```python
# Average Driving Distance Grouped by FuelType
data.groupby('fueltype')['ANNMILES'].mean()
```




    fueltype
    diesel                         11247.242260
    gas                             9620.500235
    hybrid/electric/alternative    12308.237213
    Name: ANNMILES, dtype: float64



Above shows the mean annual mileage of vehicles of different fuel types. As can be seen, the lower the fuel cost, the longer the mileage driven is.

### More Summary Statistics


```python
htype_dic = {1:  'biodiesel', 2:  'plug-in hybrid', 3:  'electric', 4:  'hybrid', -9: 'NA', -8: 'NA', -1: 'NA', 97: 'NA'}
data['hfuel'] = data['HFUEL'].map(htype_dic)
data.groupby(['fueltype','hfuel'])['ANNMILES'].mean()
```




    fueltype                     hfuel         
    diesel                       NA                11247.242260
    gas                          NA                 9620.500235
    hybrid/electric/alternative  NA                12387.113208
                                 biodiesel         12541.944444
                                 electric           8235.281720
                                 hybrid            12874.606100
                                 plug-in hybrid    11516.482587
    Name: ANNMILES, dtype: float64



Within the category of energy efficient vehicle, annual mileage ranking is : <br>
hybrid > biudiesel > plug-in hybrid > electric <br>
(Battery size probably plays a role in limiting the mileage of pure electric vehicles.) <br>


```python
area_avg_mile = data.groupby(['HHSTATE','HH_CBSA'])['ANNMILES'].mean().to_frame()
# average annual driving distance of each region (State + Core Based Statistical Area)
data = pd.merge(data,area_avg_mile, how = 'right', left_on = ['HHSTATE','HH_CBSA'], right_index = True)
# add as new column in the dataframe
data = data.rename(columns = {'ANNMILES_x': 'ANNMILES', 'ANNMILES_y': 'area_avg_mile'})
data['relative_mile'] = data['ANNMILES']/data['area_avg_mile']
# this is relative driving milage, compared to the average level of the local area
data.groupby(['fueltype','hfuel'])['relative_mile'].mean()

```




    fueltype                     hfuel         
    diesel                       NA                1.173388
    gas                          NA                0.985955
    hybrid/electric/alternative  NA                1.284829
                                 biodiesel         1.306946
                                 electric          0.888567
                                 hybrid            1.351362
                                 plug-in hybrid    1.223351
    Name: relative_mile, dtype: float64



Energy efficient vehicles are driven for more mileage compared to its local average levels.
This further comfirms the driving behavior pattern. Until this stage we have looked at the general data pattern without handling the "selection bias" issue. We will try to deal with this problem using propensity score matching.<br><br>
## Propensity Score Matching
The following section will match the treatment/control groups:
* treatment group: hybrid/electric/alternative vehicles
* control group: gasoline/diesel vehicles
They will be matched by both household and vehicle characteristics. The python package [pymatch](https://github.com/benmiroglio/pymatch) will be used here.

### Calculating Propensity Score


```python
trt_dic = {1: False, 2: False,3: True}
data['treatment'] = data['FUELTYPE'].map(trt_dic)
fields = ["income", "homeown", "urban", "vehtype", "VEHAGE", "HHSIZE", "HHSTATE", "HHVEHCNT", "treatment", "ANNMILES"] 
data_match = data[fields]
treatment = data_match[data_match.treatment == 1]
control = data_match[data_match.treatment == 0]
m = Matcher(treatment, control, yvar = "treatment", exclude = ["ANNMILES"])
```

    Formula:
    treatment ~ income+homeown+urban+vehtype+VEHAGE+HHSIZE+HHSTATE+HHVEHCNT
    n majority: 175746
    n minority: 4966



```python
np.random.seed(1)
m.fit_scores(balance = True, nmodels = 100)
```

    Fitting Models on Balanced Samples: 100\100
    Average Accuracy: 76.45%


This step calculates the **propensity score** (probability of purchasing a clean energy vehicle).<br>
The treatment group (minority) has 4966 obs, whereas control group (majority) has 175746 obs, rendering imbalanced sample. <br>
Use "balance = true" so that when fitting logistic models, equal number of treatment and control samples are used.<br>
Use "nmodels = 100" to randomly draw sample for 100 times, so that the logistic model will be fitted using more data points from the control group (the majority group).<br>
    
The **average accuracy** is 76.45%, suggesting a good fit of the model.


```python
m.predict_scores() # calculate p-score
```


```python
m.plot_scores()
```


![png](output_18_0.png)


The above step plot the p-score distribution of treatment group and control group. <br>
For each p-score, there is positive chance of belonging to either group. (**common support**) <br>
The treatment group has much higher scores than the control group. (**separability**) <br>
Such evidence support the use of propensity score matching.

### Setting Matching Threshold


```python
m.tune_threshold(method='random')
```


![png](output_21_0.png)


Tune threshold for matching:<br>
From the graph: a threshold at 0.0003 will retain 100% of the data.


```python
m.match(method = "min", nmatches = 1, threshold = 0.0003)
# matching the treatment data points with control data points
```

### Preview Matched Data


```python
m.matched_data.sort_values("match_id").head(6)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>income</th>
      <th>homeown</th>
      <th>urban</th>
      <th>vehtype</th>
      <th>VEHAGE</th>
      <th>HHSIZE</th>
      <th>HHSTATE</th>
      <th>HHVEHCNT</th>
      <th>treatment</th>
      <th>ANNMILES</th>
      <th>scores</th>
      <th>match_id</th>
      <th>record_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62500</td>
      <td>1</td>
      <td>not_urban</td>
      <td>car</td>
      <td>7</td>
      <td>2</td>
      <td>NC</td>
      <td>2</td>
      <td>1</td>
      <td>15000</td>
      <td>0.578314</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12165</th>
      <td>62500</td>
      <td>1</td>
      <td>not_urban</td>
      <td>car</td>
      <td>7</td>
      <td>2</td>
      <td>NC</td>
      <td>2</td>
      <td>0</td>
      <td>10000</td>
      <td>0.578314</td>
      <td>0</td>
      <td>12165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>112500</td>
      <td>1</td>
      <td>not_urban</td>
      <td>car</td>
      <td>12</td>
      <td>3</td>
      <td>NC</td>
      <td>7</td>
      <td>1</td>
      <td>8571</td>
      <td>0.368433</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>148095</th>
      <td>20000</td>
      <td>1</td>
      <td>urban_cluster</td>
      <td>car</td>
      <td>11</td>
      <td>2</td>
      <td>AZ</td>
      <td>5</td>
      <td>0</td>
      <td>12000</td>
      <td>0.368417</td>
      <td>1</td>
      <td>148095</td>
    </tr>
    <tr>
      <th>2</th>
      <td>112500</td>
      <td>1</td>
      <td>urban_area</td>
      <td>car</td>
      <td>3</td>
      <td>2</td>
      <td>NC</td>
      <td>2</td>
      <td>1</td>
      <td>8000</td>
      <td>0.759758</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>143971</th>
      <td>112500</td>
      <td>1</td>
      <td>urban_area</td>
      <td>car</td>
      <td>3</td>
      <td>2</td>
      <td>NC</td>
      <td>2</td>
      <td>0</td>
      <td>10500</td>
      <td>0.759758</td>
      <td>2</td>
      <td>143971</td>
    </tr>
  </tbody>
</table>
</div>



### Validity of Matching


```python
categorical_results = m.compare_categorical(return_table = True)
categorical_results
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>before</th>
      <th>after</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>urban</td>
      <td>0.0</td>
      <td>0.119381</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vehtype</td>
      <td>0.0</td>
      <td>0.752881</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HHSTATE</td>
      <td>0.0</td>
      <td>0.960683</td>
    </tr>
  </tbody>
</table>
</div>



Above results(pvalue) shows chi-square test for independence before and after matching. After matching we want this pvalue to be > 0.05, resulting in failure to reject the null hypothesis that the frequency of the enumerated term values are independent of the test and control grousps.


```python
cc = m.compare_continuous(return_table = True)
```


```python
cc
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>ks_before</th>
      <th>ks_after</th>
      <th>grouped_chisqr_before</th>
      <th>grouped_chisqr_after</th>
      <th>std_median_diff_before</th>
      <th>std_median_diff_after</th>
      <th>std_mean_diff_before</th>
      <th>std_mean_diff_after</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>income</td>
      <td>0.0</td>
      <td>0.632</td>
      <td>0.0</td>
      <td>0.441</td>
      <td>0.416377</td>
      <td>0.00000</td>
      <td>0.494024</td>
      <td>0.005524</td>
    </tr>
    <tr>
      <th>1</th>
      <td>homeown</td>
      <td>0.0</td>
      <td>0.224</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.076510</td>
      <td>-0.023737</td>
    </tr>
    <tr>
      <th>2</th>
      <td>VEHAGE</td>
      <td>0.0</td>
      <td>0.488</td>
      <td>0.0</td>
      <td>0.567</td>
      <td>-0.501484</td>
      <td>0.24207</td>
      <td>-0.581891</td>
      <td>-0.009603</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HHSIZE</td>
      <td>0.0</td>
      <td>0.116</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.018855</td>
      <td>0.048497</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HHVEHCNT</td>
      <td>0.0</td>
      <td>0.874</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>-0.176163</td>
      <td>0.011618</td>
    </tr>
  </tbody>
</table>
</div>



Tests included: Kolmogorov-Smirnov Goodness of fit Test (KS-test) and Chi-Square Distance. We want pvalues from both the KS-test and the grouped permutation of the Chi-Square distance after matching to be > 0.05, and they both are.<br><br>

# Results
Next we will assess the difference in driving distance of the treatment and control group.

Calculate the difference through linear regression:


```python
covariates = ["treatment", "income", "homeown", "VEHAGE", "HHSIZE" ,"HHVEHCNT"] 
y=m.matched_data["ANNMILES"]
X=sm.add_constant(pd.concat([m.matched_data[covariates], 
                             pd.get_dummies(m.matched_data["HHSTATE"]),
                             pd.get_dummies(m.matched_data["urban"]),
                             pd.get_dummies(m.matched_data["vehtype"])],axis=1))
linmodel = sm.OLS(y,X).fit()
print('results from propensity score matching')
print('Average Difference: %.2f' %linmodel.params.treatment)
print('standard error: %.2f' %linmodel.tvalues.treatment)
```

    results from propensity score matching
    Average Difference: 1203.45
    standard error: 4.94


The above shows that energy efficient vehicles can induce 1203 miles more of annual driving distance, and the result is statistically significant.


```python
unmatched1 = data[data['treatment']==1]['ANNMILES'].mean()
unmatched0 = data[data['treatment']==0]['ANNMILES'].mean()
print('Difference without propensity score matching: %.2f' %(unmatched1 - unmatched0))

```

    Difference without propensity score matching: 2638.11


The above is the raw difference of driving distance between two groups (without matching). This is greater than the matched result, providing evidence for selection bias: those who buy energy efficient vehicles probably drive more on average. Yet, the result from PSM indicates despite of such selection bias, buying an energy efficient vehicle will change people's driving behavior and induce more driving. <br><br>

# More analysis
Another question that policy makers might be interested in is "Does this extra mileage of energy efficient vehicle substitute some mileage from other vehicles owned by the household?".If total annual mileage of households who own energy efficient vehicles does not differ from similar households who do not own energy efficient vehicles, that might suggest subsitution between household vehicles. Therefore the following section will aggregate data to household level and conduct similar matching procedure to households.
 


```python
hh_total_mile = data.groupby(['HOUSEID'])['ANNMILES'].sum() \
    .to_frame().rename(columns = {'ANNMILES': 'hh_total_mile'})
hh_own_clean_veh = ((data.groupby(['HOUSEID'])['treatment'].sum()>0) \
    .to_frame()*1).rename(columns = {'treatment': 'hh_own_clean_veh'})
n_clean_veh = data.groupby(['HOUSEID'])['treatment'].sum() \
    .to_frame().rename(columns = {'treatment': 'n_clean_veh'})
data_hh = pd.merge(data, hh_total_mile, how = 'inner', left_on = ['HOUSEID'], right_index = True)
data_hh = pd.merge(data_hh, hh_own_clean_veh, how = 'inner', left_on = ['HOUSEID'], right_index = True)
data_hh = pd.merge(data_hh, n_clean_veh, how = 'inner', left_on = ['HOUSEID'], right_index = True)
data_hh = data_hh.drop_duplicates(['HOUSEID'], keep='first')
data_hh = data_hh[data_hh['NUMADLT']>0]
data_hh['mile_per_adult'] = data_hh['hh_total_mile']/data_hh['NUMADLT']
data_hh['num_children'] = data_hh['HHSIZE'] - data_hh['NUMADLT']
```


```python
fields = ["income", "homeown", "urban", "HHSIZE", "HHSTATE", "HHVEHCNT", "hh_own_clean_veh", "hh_total_mile","n_clean_veh", "mile_per_adult", "num_children"] 
data_match = data_hh[fields]
data_match.head()
treatment = data_match[data_match.hh_own_clean_veh == 1]
control = data_match[data_match.hh_own_clean_veh == 0]
m = Matcher(treatment, control, yvar = "hh_own_clean_veh", exclude = ["hh_total_mile","n_clean_veh","mile_per_adult"])

```

    Formula:
    hh_own_clean_veh ~ income+homeown+urban+HHSIZE+HHSTATE+HHVEHCNT+num_children
    n majority: 90974
    n minority: 4568



```python
np.random.seed(1)
m.fit_scores(balance = True, nmodels = 100)
# average accuracy: 64.91%

```

    Fitting Models on Balanced Samples: 100\100
    Average Accuracy: 64.91%



```python
m.predict_scores() # calculate p-score
m.plot_scores()
```


![png](output_40_0.png)



```python
m.tune_threshold(method='random')
```


![png](output_41_0.png)



```python
m.match(method = "min", nmatches = 1, threshold = 0.0008)
```


```python
m.matched_data.sort_values("match_id").head(6)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>income</th>
      <th>homeown</th>
      <th>urban</th>
      <th>HHSIZE</th>
      <th>HHSTATE</th>
      <th>HHVEHCNT</th>
      <th>hh_own_clean_veh</th>
      <th>hh_total_mile</th>
      <th>n_clean_veh</th>
      <th>mile_per_adult</th>
      <th>num_children</th>
      <th>scores</th>
      <th>match_id</th>
      <th>record_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62500</td>
      <td>1</td>
      <td>not_urban</td>
      <td>2</td>
      <td>NC</td>
      <td>2</td>
      <td>1</td>
      <td>20000</td>
      <td>1.0</td>
      <td>10000.000000</td>
      <td>0</td>
      <td>0.372852</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8288</th>
      <td>62500</td>
      <td>1</td>
      <td>not_urban</td>
      <td>2</td>
      <td>NC</td>
      <td>2</td>
      <td>0</td>
      <td>10000</td>
      <td>0.0</td>
      <td>5000.000000</td>
      <td>0</td>
      <td>0.372852</td>
      <td>0</td>
      <td>8288</td>
    </tr>
    <tr>
      <th>1</th>
      <td>112500</td>
      <td>1</td>
      <td>not_urban</td>
      <td>3</td>
      <td>NC</td>
      <td>7</td>
      <td>1</td>
      <td>65374</td>
      <td>1.0</td>
      <td>21791.333333</td>
      <td>0</td>
      <td>0.554536</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8861</th>
      <td>112500</td>
      <td>1</td>
      <td>not_urban</td>
      <td>3</td>
      <td>NC</td>
      <td>7</td>
      <td>0</td>
      <td>45850</td>
      <td>0.0</td>
      <td>15283.333333</td>
      <td>0</td>
      <td>0.554536</td>
      <td>1</td>
      <td>8861</td>
    </tr>
    <tr>
      <th>2</th>
      <td>112500</td>
      <td>1</td>
      <td>urban_area</td>
      <td>2</td>
      <td>NC</td>
      <td>2</td>
      <td>1</td>
      <td>20000</td>
      <td>2.0</td>
      <td>10000.000000</td>
      <td>0</td>
      <td>0.540142</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6376</th>
      <td>112500</td>
      <td>1</td>
      <td>urban_area</td>
      <td>2</td>
      <td>NC</td>
      <td>2</td>
      <td>0</td>
      <td>12800</td>
      <td>0.0</td>
      <td>6400.000000</td>
      <td>0</td>
      <td>0.540142</td>
      <td>2</td>
      <td>6376</td>
    </tr>
  </tbody>
</table>
</div>




```python
categorical_results = m.compare_categorical(return_table = True)
categorical_results
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>before</th>
      <th>after</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>urban</td>
      <td>0.0</td>
      <td>0.845934</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HHSTATE</td>
      <td>0.0</td>
      <td>0.999939</td>
    </tr>
  </tbody>
</table>
</div>




```python
cc = m.compare_continuous(return_table = True)
cc
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>ks_before</th>
      <th>ks_after</th>
      <th>grouped_chisqr_before</th>
      <th>grouped_chisqr_after</th>
      <th>std_median_diff_before</th>
      <th>std_median_diff_after</th>
      <th>std_mean_diff_before</th>
      <th>std_mean_diff_after</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>income</td>
      <td>0.0</td>
      <td>0.998</td>
      <td>0.0</td>
      <td>0.998</td>
      <td>0.856976</td>
      <td>0.0</td>
      <td>0.645195</td>
      <td>0.005165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>homeown</td>
      <td>0.0</td>
      <td>0.899</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.180183</td>
      <td>-0.003286</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HHSIZE</td>
      <td>0.0</td>
      <td>0.512</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.182328</td>
      <td>0.024008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HHVEHCNT</td>
      <td>0.0</td>
      <td>0.842</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.210103</td>
      <td>0.019599</td>
    </tr>
    <tr>
      <th>4</th>
      <td>num_children</td>
      <td>0.0</td>
      <td>0.287</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.059526</td>
      <td>0.014982</td>
    </tr>
  </tbody>
</table>
</div>




```python
covariates = ["hh_own_clean_veh", "income", "homeown", "HHSIZE" ,"HHVEHCNT"] 
linmodel = sm.OLS(m.matched_data["hh_total_mile"], 
                  pd.concat([m.matched_data[covariates], 
                             pd.get_dummies(m.matched_data["HHSTATE"]),
                             pd.get_dummies(m.matched_data["urban"])],axis=1)).fit()
print('the effect of owning a clean vehicle on household miles driven')
print('ATE:',linmodel.params.hh_own_clean_veh)
print('standard error:',linmodel.tvalues.hh_own_clean_veh)
```

    the effect of owning a clean vehicle on household miles driven
    ATE: 1729.01924946
    standard error: 4.45509371037



```python
# use "number of clean vehicle" as independent var instead
covariates2 = ["n_clean_veh", "income", "homeown", "HHSIZE" ,"HHVEHCNT"] 
linmodel2 = sm.OLS(m.matched_data["hh_total_mile"], 
                  pd.concat([m.matched_data[covariates2], 
                             pd.get_dummies(m.matched_data["HHSTATE"]),
                             pd.get_dummies(m.matched_data["urban"])],axis=1)).fit()
print('the effect of owning one more clean vehicle on household miles driven')
print('ATE:',linmodel2.params.n_clean_veh)
print('standard error:',linmodel2.tvalues.n_clean_veh)
```

    the effect of owning one more clean vehicle on household miles driven
    ATE: 1779.57031899
    standard error: 5.34001409422



```python

# use "mile per adult" as dependent var, "own clean veh dummy" as independent var
covariates3 = ["hh_own_clean_veh", "income", "homeown", "num_children" ,"HHVEHCNT"] 
linmodel3 = sm.OLS(m.matched_data["mile_per_adult"], 
                  pd.concat([m.matched_data[covariates3], 
                             pd.get_dummies(m.matched_data["HHSTATE"]),
                             pd.get_dummies(m.matched_data["urban"])],axis=1)).fit()
print('the effect of owning a clean vehicle on miles driven per adult in the household')
print('ATE:',linmodel3.params.hh_own_clean_veh)
print('standard error:',linmodel3.tvalues.hh_own_clean_veh)
```

    the effect of owning a clean vehicle on miles driven per adult in the household
    ATE: 881.540124761
    standard error: 3.80814019464


Findout from this section: 
* Households that purchase clean vehicles are driving more annual miles,which does not support the substitution effect evidence. 
* Individuals in households with clean energy vehicles are driving more miles, which is consistent with the above results. <br><br>

Combining the results, we can assert that owning an energy efficient vehicle leads to more driving, and this does not substitute the mileage of other vehicles owned by the household. The **rebound effect does exist for energy efficient vehicles**. 
