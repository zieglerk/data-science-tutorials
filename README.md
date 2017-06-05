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

# Table of content
 * ### 0-Intro
   * Scikit-learn-overview.ipynb
   * Web Mining Project .ipynb
 * ### 1-Datasets_Visualization_and_preprocessing
   * 1-IRIS.ipynb
   * 2-Boston_house_dataset.ipynb
   * 3-MNIST.ipynb
   * 4-UCI_CAR.ipynb
   * 5-20newsgroups.ipynb
   * 6-KDD_cup_2000_data_set.ipynb
   * Crawling_twitter_with_python.ipynb
   * MDS_projection.ipynb ([IRIS](./1-Datasets_Visualization_and_preprocessing/1-IRIS.ipynb))
   * PCA_projection.ipynb ([IRIS](./1-Datasets_Visualization_and_preprocessing/1-IRIS.ipynb))
   * scikit-learn-overview-and-preprocessing.ipynb ([IRIS](./1-Datasets_Visualization_and_preprocessing/2-Boston_house_dataset.ipynb))
   * VA-InformationVisualisation-with-JavaScript-and-3DJs.ipynb
 * ### 2-Clustering
   * Clustering_overview.ipynb ([IRIS](./1-Datasets_Visualization_and_preprocessing/2-Boston_house_dataset.ipynb)) ([MNIST](./1-Datasets_Visualization_and_preprocessing/3-MNIST.ipynb))
   * Tutorial_clustering_for_outlier_detection_3D.ipynb (Kddcup 1999)
   * Tutorial_clustering_for_outlier_detection.ipynb (Kddcup 1999)
 * ### 3-Association-Rules
   * Apriori_asaini.ipynb ([MBE_dataset](./3-Association-Rules/dataset/INTEGRATED-DATASET.csv))
   * Apriori.ipynb ([Boston house](./1-Datasets_Visualization_and_preprocessing/2-Boston_house_dataset.ipynb))
   * Apriori_server.ipynb ([Mango_dataset](./3-Association-Rules/dataset/data.csv))
   * Assignment_Association_rule_learning.ipynb
   * Tutorial_association_rule_learning_shopping_basket.ipynb (KDDcup 2000)
 * ### 4-Linear_regression_and_logistic_regression
   * Assignment_Linear_Regression.ipynb
   * Assignment_Logistic_regression.ipynb ([UCI_car](./1-Datasets_Visualization_and_preprocessing/4-UCI_CAR.ipynb))
   * Boston_house_Linear_Regression.ipynb ([Boston house](./1-Datasets_Visualization_and_preprocessing/2-Boston_house_dataset.ipynb))
   * Linear_regression_diabetes_dataset.ipynb
   * Linear-Regression.ipynb ([Boston house](./1-Datasets_Visualization_and_preprocessing/2-Boston_house_dataset.ipynb))
   * Logistic_regression.ipynb ([IRIS](./1-Datasets_Visualization_and_preprocessing/1-IRIS.ipynb))
   * Small_scale_linear_regression.ipynb (KDDcup)
   * Supervised_Learning_with_Linear_Models.ipynb ([Boston house](./1-Datasets_Visualization_and_preprocessing/2-Boston_house_dataset.ipynb))
 * ### 5-KNN_classification
   * KNN_classification.ipynb ([IRIS](./1-Datasets_Visualization_and_preprocessing/1-IRIS.ipynb))
   * Metrics.ipynb ([IRIS](./1-Datasets_Visualization_and_preprocessing/1-IRIS.ipynb))
 * ### 6-Bayes-Learning
   * 20news-bydate-test
   * 20news-bydate-train
   * Bayes-Learning.ipynb ([IRIS](./1-Datasets_Visualization_and_preprocessing/1-IRIS.ipynb)) ([20 news group](./1-Datasets_Visualization_and_preprocessing/5-20newsgroups.ipynb))
 * ### 7-Decision-Trees.ipynb ([UCI_car](./1-Datasets_Visualization_and_preprocessing/4-UCI_CAR.ipynb))
 * ### 8-Neural-Networks
   * keras-mnist.ipynb ([MNIST](./1-Datasets_Visualization_and_preprocessing/3-MNIST.ipynb))
   * Simple-NN.ipynb (make_moons)
   * Stacked-Denoising-Autoencoders.ipynb
 * ### 9-SVM
   * Assignment_SVM_for_OCR.ipynb ([MNIST](./1-Datasets_Visualization_and_preprocessing/3-MNIST.ipynb))
   * Support_Vector_Machines.ipynb ([IRIS](./1-Datasets_Visualization_and_preprocessing/1-IRIS.ipynb))
 * ### A-Advanced_modules
   * III.NLP-with-NLTK-Short-Intro.ipynb
 * ### B-Scripts
