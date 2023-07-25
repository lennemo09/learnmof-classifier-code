import argparse
import os
import random
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from cutmix import CutMixCollator, CutMixCriterion
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from select_gpu import get_free_gpu
import timm
from torchinfo import summary
from tqdm.auto import tqdm

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_device_index():
    # Setup device-agnostic code
    if torch.cuda.is_available():
        free_gpu_id = get_free_gpu()
        device = torch.device(f"cuda:{free_gpu_id}")
        torch.cuda.set_device(free_gpu_id)
        print(f"Using GPU {free_gpu_id}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    return device

plot_dir = './training_output/plots'
log_dir = './training_output/logs'
saved_model_dir = './training_output/weights'

parser = argparse.ArgumentParser(description='Model training client for LearnMOF project.')

device = get_device_index()

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
                        default=2, 
                        help='number of workers')
    parser.add_argument('--lr_decay',
                        type=float, 
                        default=1e-4, 
                        help='learning rate decay')
    parser.add_argument('--img_size',
                        type=int, 
                        default=512, 
                        help='image size for training')
    parser.add_argument('--seed',
                        type=int,
                        default=1337)
    parser.add_argument('--data_path',
                        type=str, 
                        default='dataset_split-70-15-15', 
                        help='path to dataset')
    parser.add_argument("--oversampled", action=argparse.BooleanOptionalAction,
                        help='use use oversampled images for training')                   
    parser.add_argument("--cutmix", action=argparse.BooleanOptionalAction,
                        help='use cutmix augmentation')
    parser.add_argument("--cutmix_alpha",
                        type=float,
                        default=1.0)
    parser.add_argument("--grayscale_random", action=argparse.BooleanOptionalAction,
                        help='use random grayscale images')
    parser.add_argument("--freeze_params", action=argparse.BooleanOptionalAction,
                        help='freeze model parameters')

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

def log_results(model_id, model_name, args, model_summary, train_accuracies, train_losses, val_accuracies, val_losses, log_dir):
    log_filename = f"model_{model_id}_log.txt"
    log_filepath = os.path.join(log_dir, log_filename)

    with open(log_filepath, 'w') as log_file:
        log_file.write(f"Model ID: {model_id}\n")
        log_file.write(f"Model Name: {model_name}\n")
        log_file.write(f"Hyperparameters:\n")
        for arg in vars(args):
            log_file.write(f"{arg}: {getattr(args, arg)}\n")
        log_file.write("\n")
        log_file.write(f"Model Summary:\n{model_summary}\n\n")
        log_file.write("Training Output:\n")
        log_file.write(f"Epoch\tTrain Loss\tTrain Accuracy\tVal Loss\tVal Accuracy\n")
        for epoch, (train_loss, train_accuracy, val_loss, val_accuracy) in enumerate(zip(train_losses, train_accuracies, val_losses, val_accuracies), 1):
            log_file.write(f"{epoch}\t{train_loss:.6f}\t{train_accuracy:.6f}\t{val_loss:.6f}\t{val_accuracy:.6f}\n")

    print(f"Results logged to: {log_filepath}")


def train_model(model, train_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)

        if isinstance(labels, (tuple, list)):
            labels1, labels2, lam = labels
            labels = (labels1.to(device), labels2.to(device), lam)
        else:
            labels = labels.to(device)                         

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if isinstance(labels, (tuple, list)):
            _, preds = torch.max(outputs, dim=1)
            y1, y2, lam = labels
            num = inputs.size(0)
            correct1 = preds.eq(y1).sum().item()
            correct2 = preds.eq(y2).sum().item()
            correct_predictions = (lam * correct1 + (1 - lam) * correct2) / num

            epoch_accuracy = correct_predictions / inputs.size(0)
        else:
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            epoch_accuracy = correct_predictions / total_samples
    epoch_loss /= len(train_loader)

    return epoch_accuracy, epoch_loss

def validate_model(model, val_loader, criterion):
    model.eval()
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.inference_mode():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device)

            if isinstance(labels, (tuple, list)):
                labels1, labels2, lam = labels
                labels = (labels1.to(device), labels2.to(device), lam)
            else:
                labels = labels.to(device) 

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            
            if isinstance(labels, (tuple, list)):
                _, preds = torch.max(outputs, dim=1)
                y1, y2, lam = labels
                num = inputs.size(0)
                correct1 = preds.eq(y1).sum().item()
                correct2 = preds.eq(y2).sum().item()
                correct_predictions = (lam * correct1 + (1 - lam) * correct2) / num

                epoch_accuracy = correct_predictions / inputs.size(0)
            else:
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                epoch_accuracy = correct_predictions / total_samples
    epoch_loss /= len(val_loader)

    return epoch_accuracy, epoch_loss


def train_and_validate_model(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

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

    return model, train_accuracies, train_losses, val_accuracies, val_losses
    

def main():
    initialise_cli_arguments()
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed) 

    seed_everything(args.seed)

    model_name = args.model
    num_epochs = args.epochs
    image_size = args.img_size
    
    # 1. Load model
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=3
    )

    # 2. Load dataset 
    data_path = args.data_path
    # Setup train and testing paths
    if args.oversampled:
        train_dir = data_path + "/" + "train_oversampled"
    else:
        train_dir = data_path + "/" + "train"
    val_dir = data_path + "/" + "val"

    # 3. Data augmentation
    train_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=args.img_size)
    ]

    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=args.img_size)
    ]

    if args.grayscale_random:
        train_transforms.append(transforms.RandomGrayscale(p=0.3)),
        test_transforms.append(transforms.RandomGrayscale(p=0.3))

    if args.cutmix:
        collator = CutMixCollator(args.cutmix_alpha)
        criterion = CutMixCriterion(reduction='mean')
    else:
        collator = torch.utils.data.dataloader.default_collate
        criterion = nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.lr_decay)
    
    # Write transform for image
    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    print('Transforms to apply:',train_transforms)
          
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=train_transforms, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    val_data = datasets.ImageFolder(root=val_dir,
                                    transform=test_transforms)
    
    class_names = train_data.classes

    print(f"Train data:\n{train_data}\nVal data:\n{val_data}")

    print('Classes', class_names)

    run_batch_size = args.batch_size

    # 4. Prepare dataloaders
    print(f"Creating DataLoader's with batch size {run_batch_size} and {args.num_workers} workers.")
    train_dataloader = DataLoader(train_data,
                                batch_size=run_batch_size,
                                shuffle=True,
                                collate_fn=collator,
                                num_workers=args.num_workers)

    val_dataloader = DataLoader(val_data,
                                batch_size=run_batch_size,
                                shuffle=False,
                                collate_fn=collator,
                                num_workers=args.num_workers)

    model = model.to(device)

    model_summary = str(summary(model, input_size=[1, 3, args.img_size, args.img_size], verbose=0))
    print(model_summary)

    model_id = find_next_model_id(saved_model_dir)

    # 5. Prepare logging
    
    # 6. Train model
    model, train_accuracies, train_losses, val_accuracies, val_losses = train_and_validate_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs)
    
    # 7. Save model and logs
    save_model(model, model_id, model_name, num_epochs, image_size, saved_model_dir)
    plot_results(train_accuracies, train_losses, val_accuracies, val_losses, model_name, image_size, plot_dir)
    log_results(model_id, model_name, args, model_summary, train_accuracies, train_losses, val_accuracies, val_losses, log_dir)
    

if __name__ == '__main__':
    main()