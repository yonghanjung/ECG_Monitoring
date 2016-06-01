# Detection of Premature Ventricular Contraction using Wavelet-based Statistical ECG monitoring
Last update : 160601

You may only consdier the folder named 'Final_Current_Working'. Other folders are temporaily used. I call this folder as 'Final' from now.
For running this program, following Python libraries should be installed: 

* numpy 
* scipy
* pywt 
* sklearn 

In the folder 'Final', there are one subfolder 'Class' and two python file 'main.py', and 'methods.py'. Details are following.

## Class - Class_SDA.py
In 'Class_SDA.py', the algorithm for constructing sparse discriminant vector, suggested in 'Sparse Discriminant Analysis (2012), Clemmensen et al (Technometrics)' are implemented. 

## methods.py 
In 'methods.py', all functions used in 'main.py' are set.  

Functions in 'methods.py' are 

 * __Loading_ECG__: Loading the ECG record 
 * __Loading_R_Peak_and_Label__: Loading annotation file (R_peak_index and labels of ECG beats are in annotation file)
 * __SoftThreshold__: Implementing the soft threshold for the universal thresholding
 * __Segmenting_ECG_Beat__: Segmenting ECG record beat by beat. 
 * __Wavelet_Transformation__: Implementing discrete wavelet transformation to each ECG beat after denoising using universal threshold.
 * __Constructing_SDA_Vector__: Constructing sparse discriminant vector as suggested in Sparse Discriminant Analysis (Technometrics) by Clemmensen (2012)
 * __Projecting_Lower_Dimensional_Vec__: Implementing low dimensional projection of vector using sparse discriminant vector
 * __Projecting_Low_Dimensional_Cov__: Implementing low dimensional projection of covariance matrix using sparse discriminant vector
 * __Constructing_T2_Stat__: Compute T2 statistics from low dimensional projected wavelet coefficients, using projected average and covariance matrix
 * __Computing_UCL__: Computing upper control limit (UCL)
 * __Evaluating_Performance_SPM__: Counting right and wrong classification result for evaluating performance of wavelet-based SPM
 * __Evaluating_Performance_SVM_NN__: Counting right and wrong classification result for evaluating performance of SVM and neural network

## main.py
'main.py' file loads functions in methods, and implement wavelet-based SPM for detecting PVC beats.

Procedures are: 

1. Reading ECG records and segmenting by beats
2. Constructing training set (initial 5min are segmented as training set) 
3. Applying wavelet transformation to each ECG beats in training and test set
4. Constructing sparse discriminant vector and projecting to the low dimensional space
5. Computing T2 statistics
6. Evaluating accuracy by counting right and wrongly classified beats
