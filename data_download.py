import kagglehub
import shutil

# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")

print("Path to dataset files:", path)

# Move to data folder
shutil.move(path, "./data")