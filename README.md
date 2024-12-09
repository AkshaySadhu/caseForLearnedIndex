**REQUIREMENTS** 
1. You will need Python 3.9+ installed on your system
2. To install the packages required run ```pip install -r src/requirements.txt```

**DATASET GENERATION**

**LATITUDE DATASET**
1. Currently we have a **dummyLatitudes.csv** since we could not upload the actual latitude files to GitHub because of size. I have listed down the steps to be followed here in detail, so as to get the full latitude dataset- https://docs.google.com/document/d/1pv9O2f0Sw5XnlwrvZzIOZe4R2oZZMNV2aKvEl1tGUQQ/edit?usp=sharing.
2. We have also authored the **osmParser.py** script, that needs to be run after following the instructions in the doc.
3. If you followed the instructions of the doc, you have the actual datasets. You would then need to change the csv file name in all the .py and .ipynb files (we have currently used **dummyLatitudes.csv** so that code can be run without following step 1 too).

**LOGNORMAL DATASET**
1. We have included **dummyLognormal.csv**, since we could not upload the actual lognormal data due to size. In order to get the full file (**lognormalSortedData.csv**), run the **lognormalGenerator.py**, and replace the occurrence of **dummyLognormal.csv** in each file with **lognormalSortedData.csv**.
Steps to run the code:

**ML MODELS**


**1. B-Trees:**  \
      1. B-Trees on Lognormal Data: Run **btree.py**. We will always get a 0 error on B-Trees implementation for this. This gives you the exact value of the index for the input key. \
      2. B-Trees for Latitude data: Run **btree_coordinates.py**. \
      
**2. Linear Regression:**  \
      1. For the first version of linear regression, where we had 3 different linear models based on the range of the keys. Run **linearRegression.py** for lognormal data. This trains on the full dataset and tests on 10k, 100k and 1M keys and plots predicted vs ground truth graphs for the test. To get the same plots for the latitude dataset run **linearRegression_coordinates.py**. \
      2. Piecewise polynomial: Run the **linearRegression_rms.py**. This runs the **fit_piecewise_polynomial(X, y, ranges, max_rmse)** function to get the perfect split for linear equations to get a piecewise polynomial which achieves the specified max_rmse (max root mean squared error). \
      
**3. Neural Networks:** \
      1. 2 layer fully-connected Neural network: **NN.py** \
      2. Statistics on training, lookup time and model size **stats_gather.py** \
      3. Please change filenames to **dummyLognormal.csv** and **dummyLatitudes.csv** as required \

**4. Hybrid Index:**   \
    1. You may run **hybridindex_latitude.py** to plot the scatter plot against the latitude dataset, using the hybrid index. For the latitude dataset, we observed that the mix of linear regression model as the root model along with 20 neural network models in the second layer yielded best results.\
    2. You may run the **hybridindex_lognormal.py** to plot the scatter plot against the lognormal dataset, using the hybrid index. For the lognormal dataset, we observed that the mix of random forest regression model as the root model along with 20 neural network models in the second layer yielded best results.\
    You may test against the number of neural network models (by changing the **num_segments** variable in the code) in the second layer for comparitive analysis.
    

**5. Spline Interpolation:**    \
    1. Basic Cubic Spline: You can run **splineInterpolation.py** to get a basic fit (and visualisation) of a basic cubic spline interpolation, where each point is a knot. \
    2. Cubic Spline with Metrics (lookup time, model space, training time): Run **splineInteroplation2.py**. Here too all points are used as knots \
    3. Spline interpolation with fewer knots: Run **splineInterpolation3.py**. \
    4. Spline interpolation with lognormal data: Run **splineInterpolationWithLognormal.py**.
