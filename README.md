# RecSys_challenge
A music recommender system as part of Kaggle Championship. 

# Data 
train.csv : the training set describing which tracks are included in the playlists
tracks.csv : supplementary information about tracks
target_playlists.csv : the set of target playlists that will receive recommendations
train_sequential.csv: list of ordered interactions for the 5k sequential playlists. You can find the file here.
sample_submission.csv: correct format for submissions

## train.csv
playlist_id : identifier of the playlist
track_id : identifier of the track

# Process 

- Sparse matrices creation
- Recommender creation ( Type of recommender as to be chosen ) 
- Evaluation of parameters of the extracted test set of train.csv 
- Choice of parameters according to the previous evaluation

## Environment 
- Conda

## Packages 
- sklearn 
