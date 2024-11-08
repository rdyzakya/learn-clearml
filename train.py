import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from read_data import MnistDataloader
from tqdm import tqdm
from argparse import ArgumentParser

from clearml import Task, Logger
task = Task.init(project_name='test-clearml', task_name='training')

# Define a custom dataset class for your numpy data
class NumpyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training settings

def init_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training")
    args = parser.parse_args()
    return args

def train(batch_size, epochs, learning_rate, params):
    # Create dataset and split into training and validation sets
    input_path = './data/1'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    train_dataset = NumpyDataset(x_train, y_train)
    val_dataset = NumpyDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with validation
    train_pbar = tqdm(total=epochs * len(train_loader), desc="Training")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.update(1)
        
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {running_loss / len(train_loader):.4f}")
        Logger.current_logger().report_scalar(
        title="loss", series="training", iteration=epoch, value=running_loss / len(train_loader)
        )
        
        # Validation phase
        eval_pbar = tqdm(total=len(val_loader), desc=f"Evaluation epoch {epoch + 1}")
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                eval_pbar.update(1)
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        Logger.current_logger().report_scalar(
            title="loss", series="validation", iteration=epoch, value=val_loss
        )
        Logger.current_logger().report_scalar(
            title="accuracy", series="validation", iteration=epoch, value=accuracy
        )

    print("Training complete.")

    # Directory to save the model
    save_dir = './models'
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, 'simple_cnn.pth'))
    print("Model saved successfully.")
    return model

if __name__ == "__main__":
    args = init_args()
    dummy_params = {
        "n_layers" : 3,
        "n_units" : 64,
    }
    task.connect(dummy_params)
    config_file_yaml = task.connect_configuration(
        name="config",
        configuration="dummy_config.yaml",
    )
    print(dummy_params)
    model = train(args.batch_size, args.epochs, args.learning_rate, dummy_params)

    # using pytorch already automatically upload the artifact
    task.upload_artifact(name="model", artifact_object="./models/simple_cnn.pth")