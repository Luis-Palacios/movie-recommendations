# Recommender system for movies
Repository to demonstrate how to use machine learning to generate recommendations. It's use movie lens data.
This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org),
a movie recommendation service. It contains 100004 ratings and 1296 tag applications across 9125 movies.

It's uses [Surprise](http://surpriselib.com/) "a Python [scikit](https://www.scipy.org/scikits.html) building and analyzing recommender systems".

## Requirements
1. [Python](https://www.python.org/downloads/)
2. [Anaconda](https://www.anaconda.com/download/)

## Project setup
1. Clone this repository and enter the repository directory
2. Create your enviroment using:
`conda env create -f environment.yml`
3. Activate your enviroment:
    ### Windows
    `activate movie-ratings`
    ### Linux or OSX
    `source activate movie-ratings`
4. Run Jupyter notebook `jupyter notebook`


See [Movie-Recommendations notebook](Movie-Recommendations.ipynb) for more details
