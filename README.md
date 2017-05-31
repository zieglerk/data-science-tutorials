# data-science-tutorials

<!-- 4.[Unsupervised Learning: Projections and Manifolds](./3-Association-Rules/Apriori_server.ipynb)
 -->
 
This repository contains jupyter-notebooks to accompany the tutorials for our data science lectures. The following topics are covered (each within a separate folder).

1. Dataset Visualization (Boston Housing minus the linear regression;
also other datasets like Flower, MNIST-digits, 20newsgroups) working/visualizing one dataset (incl. Matplotlib; .describe attribute; box-plot, min-max-normilization; boston housing; linear reg c/o dsP)
2. Clustering
3. Association Rule Learning (dataset yet to be determined; preferably from scikit learn)
4. Regression (linear regression from Boston Housing and Car Prices)
5. Bayes Learning (for spam filtering/text classification) 
6. Classification with Decision Trees (start with small 5-line dataset)
7. Neural Networks (use keras.io to build a neural network for
MNIST-digit classification) keras (for MNIST class); OPT gensim (for
	word2vec; pick dataset from tensorflow); then auto-encoder for
	representatino learning
8. OPTIONAL MapReduce

# Packages

See our [python-tutorials](https://github.com/zieglerk/python-tutorials) on instructions how to set this up on your
machine.

## required

- Python (>= 2.7 or >= 3.3)
- NumPy (>= 1.6.1)
- SciPy (>= 0.9)
- scikit-learn (>=0.18.1);
  [documentation](http://scikit-learn.org/stable/documentation.html),
  also as
  [pdf](http://scikit-learn.org/dev/_downloads/scikit-learn-docs.pdf)
  with [Quick
  Start](http://scikit-learn.org/stable/tutorial/basic/tutorial.html) and
  [Tutorials](http://scikit-learn.org/stable/tutorial/)
- Matplotlib >= 2.1.1

## optional

- Pandas; [documentation] also as
  [pdf](http://pandas.pydata.org/pandas-docs/version/0.18.1/pandas.pdf)

## Table of content
 * ### 0-Intro
   * Scikit-learn-overview.ipynb
   * Web Mining Project .ipynb
 * ### 1-Datasets_Visualization_and_preprocessing
   * Boston_house_dataset.ipynb
   * Crawling_twitter_with_python.ipynb
   * dataset
   * Datasets-and-Visualizations.ipynb
   * KDD_cup_2000_data_set.ipynb
   * MDS_projection.ipynb
   * PCA_projection.ipynb
   * scikit-learn-overview-and-preprocessing.ipynb
   * VA-InformationVisualisation-with-JavaScript-and-3DJs.ipynb
 * ### 2-Clustering
   * Clustering_overview.ipynb(MNIST)([IRIS](./1-Datasets_Visualization_and_preprocessing/Datasets-and-Visualizations.ipynb))
   * Tutorial_clustering_for_outlier_detection_3D.ipynb
   * Tutorial_clustering_for_outlier_detection.ipynb
 * 3-Association-Rules
   * Apriori_asaini.ipynb
   * Apriori.ipynb
   * Apriori_server.ipynb
   * Assignment_Association_rule_learning.ipynb
   * dataset
   * Tutorial_association_rule_learning_shopping_basket.ipynb
 * 4-Linear_regression_and_logistic_regression
   * Assignment_Linear_Regression.ipynb
   * Assignment_Logistic_regression.ipynb
   * Boston_house_Linear_Regression.ipynb
   * dataset
   * Linear_regression_diabetes_dataset.ipynb
   * Linear-Regression.ipynb
   * Logistic_regression.ipynb
   * Small_scale_linear_regression.ipynb
   * Supervised_Learning_with_Linear_Models.ipynb
 * 5-KNN_classification
   * KNN_classification.ipynb
   * Metrics.ipynb
 * 6-Bayes-Learning
   * 20news-bydate-test
   * 20news-bydate-train
   * Bayes-Learning.ipynb
 * 7-Decision-Trees.ipynb
 * 8-Neural-Networks
   * 6-keras-mnist.ipynb
   * keras-mnist.ipynb
   * pics
   * Simple-NN.ipynb
   * Stacked-Denoising-Autoencoders.ipynb
 * 9-SVM
   * Assignment_SVM_for_OCR.ipynb
   * Support_Vector_Machines.ipynb
 * A-Advanced_modules
   * III.NLP-with-NLTK-Short-Intro.ipynb
 * B-Scripts
   * plot_classification.py
   * plot_cluster_comparison.py
   * plot_iris_logistic.py
   * plot_kmeans_digits.py
   * plot_svm_margin.py
 * README.md
 * script.sh