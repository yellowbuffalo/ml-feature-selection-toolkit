# ml-feature-selection-toolkit
The toolkit is for feature selection in data modeling, implement [sklearn RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) to compute feature importances for training data and provide the advice for feature selection.
* Input: X features, Y feature(Ground Truth for Classification)
* Output: Importances ranking for features.
* Using RandomForest train the data and doing feature selection.

## Usage:
* Using following to build environment
  ```console
  user@bar:~$ python3 -m venv myenv
  user@bar:~$ source myenv/bin/activate
  user@bar:~$ pip3 install -r requirements.txt
  user@bar:~$ python3 example.py
  ```
* The simple usage will like code below:
  ```python
  
  wine = datasets.load_wine() # Using wine dataset for example
  # Split data to X and y
  X = wine.data 
  y = wine.target

  extractor = Extractor(X,y) # Call the Extractor class and input X and y.
  parameters = {'min_samples_split':[2,3,4], 'min_samples_leaf':[1,2,3]} # parameters for grid search(Optional)
  extractor.gridSearch(parameters)
  extractor.training() # Feature selection
  print(extractor.feature_importances) # Output the features importances
  ```
## Result for example:
![result](https://github.com/yellowbuffalo/ml-feature-selection-toolkit/blob/main/plot/result.png?raw=true)
