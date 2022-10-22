import copy
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from featureExtractor import *
import sklearn
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

"""
This is the example for featureExtractor, and using the wine dataset.
 1. call the Extractor class.
 2. Setting parameters to be grid search.(Optional)
 3. Training by sklearn RandomForestClassifier.
 4. Get and plot the feature importance.
"""

# Loadning dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Call the Extractor class
extractor = Extractor(X,y)
# Setting the parameters want to be used in the grid search method.(Customizable)
parameters = {'min_samples_split':[2,3,4], 'min_samples_leaf':[1,2,3]}
extractor.gridSearch(parameters) # Conduct grid search(Optional)
extractor.training() # Training and get the feature importance.

feature_importances = pd.DataFrame({
    'feature_name': wine.feature_names,
    'importance':extractor.feature_importances
})
# Sort and list the importances for every feature.
feature_importances_top = feature_importances.sort_values(['importance'], ascending=False).reset_index(drop=True)
print(feature_importances_top)

# Plot the result
sns.set(font_scale = 2,style="whitegrid")
plt.figure(figsize=(40, 15))

# Make a barplot
p = sns.barplot(
    x="importance", 
    y="feature_name", 
    data=feature_importances_top, 
    estimator=sum,
    color='#0B346E'
)
p.set_xlabel("Feature importances score", fontsize = 30, weight='bold')
p.set_ylabel("", fontsize = 30)
p.set_title("Feature Importance", fontsize = 40, weight='bold')
fig = p.get_figure()
fig.savefig("./plot/result.png", dpi=300) # Save the result as png.