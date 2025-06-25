from correlation import correlation_analysis
from preprocess import preprocess_data
from supervised_task import supervised_task
from data_clustering import data_clustering

# Linking all the code

# preprocessed data is saved in the processed folder
preprocess_data()

# Results are saved in correlation_results folder
correlation_analysis('./processed/full_preprocessed.csv')

# Results are saved in supervised_task_results folder
supervised_task()

# Results are saved in the data-clustering-results folder
data_clustering()

print('All tasks completed successfully!')
