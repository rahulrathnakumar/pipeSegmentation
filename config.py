# config file for training 

# define experiment details as a multiline string

experiment_details = """
This experiment uses w/ MCD model to perform segmentation w/ RGB data.
"""

configDict = {
    'batch_size': 32,
    'root_dir': 'data/',
    'train_txt': 'train.txt',
    'val_txt': 'val.txt',
    'num_workers': 0,
    'num_epochs': 500,
    'learning_rate': 1e-2,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'scheduler_step_size': 100,
    'scheduler_gamma': 0.9,
    'network': 'BayesianSegmentationNetwork',
    'experiment_name': 'rgbSegmentation_' + 'BayesianSegmentationNetwork_MCD',
    'log_interval': 10,
    'gpu_id': 0,
    'Experiment details': experiment_details
}
