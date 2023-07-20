import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from select_gpu import get_free_gpu

plot_dir = './training_output/plots'
log_dir = './training_output/logs'
saved_model_dir = './training_output/weights'

parser = argparse.ArgumentParser(description='Model training client for LearnMOF project.')

def initialise_cli_arguments():
    parser.add_argument('--model',
                        type=str,
                        default='efficientnetv2_s',
                        help='model architecture')
    parser.add_argument('--lr',
                        type=float, 
                        default=1e-3, 
                        help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='number of epochs')
    parser.add_argument('--batch_size',
                        type=int, 
                        default=16, 
                        help='batch size')
    parser.add_argument('--num_workers',
                        type=int, 
                        default=8, 
                        help='number of workers')
    parser.add_argument('--lr_decay',
                        type=float, 
                        default=1e-4, 
                        help='learning rate decay')
    parser.add_argument('--img_size',
                        type=int, 
                        default=512, 
                        help='image size for training')
    parser.add_argument('--data_path',
                        type=str, 
                        default='dataset_split-70-15-15', 
                        help='path to dataset')
    parser.add_argument("--oversampled", action=argparse.BooleanOptionalAction,
                        help='use use oversampled images for training')                   
    parser.add_argument("--cutmix", action=argparse.BooleanOptionalAction,
                        help='use cutmix augmentation')
    parser.add_argument("--grayscale", action=argparse.BooleanOptionalAction,
                        help='use grayscale images')

# Function to find the next available model ID
def find_next_model_id(saved_model_dir):
    model_ids = [int(file_name.split('_')[0]) for file_name in os.listdir(saved_model_dir) if file_name.endswith('.pth')]
    return max(model_ids, default=0) + 1

def save_model(model, model_id, model_name, epochs, image_size, saved_model_dir):
    model_filename = f"{model_id}_{model_name}_{epochs}_{image_size}.pth"
    model_filepath = os.path.join(saved_model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)
    print(f"Model saved to: {model_filepath}")

def plot_results(train_accuracies, train_losses, val_accuracies, val_losses, model_name, image_size, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    epochs = len(train_accuracies)
    x_vals = list(range(1, epochs + 1))

    axes[0].plot(x_vals, train_losses, label='Train Loss')
    axes[0].plot(x_vals, val_losses, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Losses')
    axes[0].legend()

    axes[1].plot(x_vals, train_accuracies, label='Train Accuracy')
    axes[1].plot(x_vals, val_accuracies, label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracies')
    axes[1].legend()

    plt.suptitle(f"{model_name} - Image Size: {image_size}")
    plt.tight_layout()

    plot_filename = f"{model_name}_{image_size}_plot.png"
    plot_filepath = os.path.join(plot_dir, plot_filename)
    plt.savefig(plot_filepath)
    print(f"Plot saved to: {plot_filepath}")
    plt.close()

def train_and_validate_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, model_name, image_size, saved_model_dir, plot_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model_id = find_next_model_id(saved_model_dir)

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_accuracy, epoch_train_loss = train_model(model, train_loader, optimizer, criterion)
        train_accuracies.append(epoch_train_accuracy)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_accuracy, epoch_val_loss = validate_model(model, val_loader, criterion)
        val_accuracies.append(epoch_val_accuracy)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")

    save_model(model, model_id, model_name, num_epochs, image_size, saved_model_dir)
    plot_results(train_accuracies, train_losses, val_accuracies, val_losses, model_name, image_size, plot_dir)


def main():
    initialise_cli_arguments()
    parser.parse_args()

if __name__ == '__main__':
    main()