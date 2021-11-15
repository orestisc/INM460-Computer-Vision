import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')
from dataset import DatasetLoader
from model import Net
import CONFIG

image_size = CONFIG.IMAGE_SIZE
use_gpu = torch.cuda.is_available()

def train(model, train_loader, test_loader, start_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if use_gpu:
        criterion = criterion.cuda()
        model = model.cuda()

    model.train()
    accuracy = "N/A"
    loss = "N/A"
    print("--------------------------------------------------------")
    print(f"Training with {CONFIG.NUM_EPOCHS} epochs, 0.001 learning rate and {CONFIG.BATCH_SIZE} batch size.")
    for epoch_index in range(start_epoch, CONFIG.NUM_EPOCHS):
        running_loss = 0

        # train loop
        for episodic_index, (images, targets) in enumerate(train_loader):

            if use_gpu:
                images = images.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            score = model(images)
            loss_ = criterion(score, targets)

            loss_.backward()
            optimizer.step()

            running_loss += loss_.item()
        loss = running_loss/len(train_loader)
        print(f"Epoch: {epoch_index + 1} | Loss: {loss}")
        if (epoch_index % CONFIG.MODEL_SAVE_EPOCHS == 0) or (epoch_index == CONFIG.NUM_EPOCHS - 1):
            if epoch_index == 0:
                continue
            print("Evaluating model...")
            model.eval()
            with torch.no_grad():
                total_predictions, correct_predictions = eval(model, test_loader)
                accuracy = round(100 * correct_predictions/total_predictions, 2)
            print(f"Epoch: {epoch_index + 1} | Loss: {loss} | Accuracy: {accuracy}")
    # Saving model
    print("--------------------------------------------------------")
    print("Training complete")
    print("--------------------------------------------------------")
    print(f"model tested on {total_predictions} samples. Accuracy: {accuracy}%")
    print("Saving model...")
    outfile = os.path.join(CONFIG.CHECKPOINT_DIR, 'checkpoint_epoch_{:d}.tar'.format(epoch_index))
    torch.save({'epoch': epoch_index, 'state': model.state_dict()}, outfile)
    print("Model Saved")

def eval(model, test_loader):
    # getting predictions on test set and measuring the performance
    correct_predictions, total_predictions = 0, 0
    for images, labels in test_loader:
        for i in range(len(labels)):
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()

            img = images[i].view(1, 3, image_size[0], image_size[1])
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.cpu()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.cpu()[i]
            if(true_label == pred_label):
                correct_predictions += 1
            total_predictions += 1
    return total_predictions, correct_predictions

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize(image_size),
            transforms.PILToTensor()
        ]),
    }

    train_dataset = DatasetLoader(CONFIG.PATH_TO_DATASET_DIR, data_transforms, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        shuffle=True
    )

    test_dataset = DatasetLoader(CONFIG.PATH_TO_DATASET_DIR, data_transforms, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        shuffle=True
    )

    # display some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))

    model = Net(num_classes=CONFIG.NUM_CLASSES)

    start_epoch = 0
    if CONFIG.RESUME:
        assert os.path.exists(CONFIG.CHECKPOINT_PATH), "Checkpoint not found"
        print("Checkpoint to resume: ", CONFIG.CHECKPOINT_PATH)
        tmp = torch.load(CONFIG.CHECKPOINT_PATH)
        start_epoch = tmp['epoch'] + 1
        print("Restored epoch is", tmp['epoch'])
        state = tmp['state']
        model.load_state_dict(state)
        print("Model Loaded !")

        # print(model.state_dict()["epoch"])
    train(model, train_loader, train_loader, start_epoch)


if __name__ == "__main__":
    main()
