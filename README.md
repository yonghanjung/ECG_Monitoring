# Research : ECG Monitoring
Last update : 150528

## Module in order 
* <code> Data_Preparation </code>
: Load Data and Label file 
* <code> Segment </code>
: Segment into the fixed length window batch 
* <code> Wavelet_Analysis </code>
: Extract Wavelet coefficients 
* <code> Training_Set </code>
: Construct the training set 
* <code> Compute_Fisher_Score </code> 
: Compute Fisher score to extract the most seperating wavelet coefficients 
* <code> Monitor_Stat </code>
: Constructing the monitoring statistics with selected features
* <code> Statistics </code>
: Estimate the variance 
: Assumed no correlation between beats' measurement error --> Diagonal variance assumption was made


## Issue 
* Proper selection of Wavelet basis funtion 
* I choose 'db8' as the most proper wavelet basis function

## Future work 
* (Done) estimating variance
* (Done) Reduce the dimension of the covariance matrix 
* Fisher LDA applied before Fisher score selection
