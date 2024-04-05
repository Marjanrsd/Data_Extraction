# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 18:57:08 2023

@author: Marjan
"""

import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

#import the data
df= pd.read_csv(r'D:\lizzzz\factorRoute.csv')

df=df[df["Chronic stress"].notna()]
df=df[df["Percieved stress"].notna()]
df=df[df["Spatial anxiety"].notna()]
df=df[df["Trait-Anxiety"].notna()]

# df.info() understanding the structure of the data
# df.head() returns the first 5 rows of the DataFrame by default
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
print(chi_square_value)
print(f"{p_value:.3f}")

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)
print(kmo_all)
print(kmo_model)


# Create factor analysis object and perform factor analysis
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer(rotation = None, impute = "drop",n_factors=df.shape[1])
fa.fit(df)
ev,_ = fa.get_eigenvalues()
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigen Value')
# plt.grid()

#loadings
fa = FactorAnalyzer(n_factors=1,rotation='varimax')
fa.fit(df)
print(pd.DataFrame(fa.loadings_,index=df.columns))

#variance
print(pd.DataFrame(fa.get_factor_variance(),index=['Variance',
'Proportional Var','Cumulative Var']))

#communalities
print(pd.DataFrame(fa.get_communalities(),
                   index=df.columns,columns=['Communalities']))


# I wanna measure multicolinearity in the stress and anxiety measures
# VIF above 5 indicates a high multicollinearity
import pandas as pd 
# the dataset
# printing first few rows
print(df.head())
from statsmodels.stats.outliers_influence import variance_inflation_factor
# the independent variables set
X = df[['Chronic stress', 'Percieved stress', 'Spatial anxiety',
          'Trait-Anxiety']]
  
# VIF dataframe
vif_df = pd.DataFrame()
vif_df["feature"] = X.columns
  
# calculating VIF for each feature
vif_df["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print(vif_df)

#correlation matrix
# Save correlations here:
corrs=df.corr()
# Plot heatmap here:
sns.heatmap(corrs, xticklabels=corrs.columns, yticklabels=corrs.columns, 
            vmin=-1, vmax=1, center=0, cmap="PuOr", annot=True)
plt.show()
















