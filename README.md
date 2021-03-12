# ECE-143 Final Project: Joke recommendation system

This repository contains the implementation of Team 25's final project: Joke Recommendation System. 

## Dataset 

[Jester](http://eigentaste.berkeley.edu/dataset/) dataset for collaborative filtering research, AUTOLab, UC Berkeley

## Dependencies

* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [scipy](https://www.scipy.org/)
* [imageio](https://pypi.org/project/imageio/) 
* [matplotlib](https://matplotlib.org/) 
* [seaborn](https://seaborn.pydata.org/)

## Code organization
```
├── visualizations.ipynb         : Jupyter Notebook containing all visualizations shown in the presentation 
├── main.py           	         : Main file that runs both UBCF and content-based recommenders for a test user 
├── data_cleaning.py             : Contains code to clean data
├── normalization.py             : Contains code to normalize data 
├── ubcf.py                      : Contains code for the user-based collaborative filtering recommender 
└── content_based_recommender.py : Contains code for the content-based recommender 
```

## Setup and Running Instructions

1. Clone this repository   
2. Run the command: `python main.py` to see the UBCF and content-based recommender results for a test user. 
3. For visualizations, open `visualizations.ipynb` and execute the notebook. 

## References 

[Eigentaste](https://goldberg.berkeley.edu/pubs/eigentaste.pdf): A Constant Time Collaborative Filtering Algorithm. Ken Goldberg, Theresa Roeder, Dhruv Gupta, and Chris Perkins. Information Retrieval, 4(2), 133-151. July 2001.