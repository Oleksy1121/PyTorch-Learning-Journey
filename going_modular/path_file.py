from pathlib import Path

data_dir = Path('data')
images_dir = data_dir / 'pizza_steak_sushi'

train_dir = images_dir / "train"
test_dir = images_dir / "test"

models_dir = 'models'
model_dir_prefix = 'model_'
results_filename = 'results.csv'
