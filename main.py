import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from capsnet import CapsNet, CapsuleLoss
import config
from torch.utils.data import ConcatDataset
import manager_datasets as datasets_control



# Check cuda availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def get_data_loader_A_APTOS(datasets, size):

    test_transform = transforms.Compose([
                                         transforms.Grayscale(num_output_channels=1),
                                         transforms.Resize([size, size], transforms.InterpolationMode("bicubic")),
                                         transforms.ToTensor(),
                                         #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         transforms.Normalize((0.5), (0.5))
                                         ])


    dataset_list = []

    for key in datasets.keys():
        datasets[key][0]["level"].replace({2: 1, 3: 1, 4: 1}, inplace=True)
        datasets[key][0].reset_index(inplace=True)
        dataset_list.append(datasets_control.dataset(datasets[key][0], f'{datasets[key][1]}', image_transform=test_transform))


    dataset_full = ConcatDataset(dataset_list)
    train = DataLoader(dataset_full, batch_size=config.BATCH, shuffle=True, generator=torch.Generator().manual_seed(1))

    return train




def main(train_loader, test_loader):
    # Load model
    model = CapsNet().to(device)
    criterion = CapsuleLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    # Train
    EPOCHES = 50
    model.train()
    for ep in range(EPOCHES):
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
            logits, reconstruction = model(images)

            # Compute loss & accuracy
            loss = criterion(images, labels, logits, reconstruction)
            correct += torch.sum(
                torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss
            loss.backward()
            optimizer.step()
            print('Epoch {}, batch {}, loss: {}, accuracy: {}'.format(ep + 1,
                                                                      batch_id,
                                                                      total_loss / batch_id,
                                                                      accuracy))
            batch_id += 1
        scheduler.step(ep)
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))

    # Eval
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        # Add channels = 1
        images = images.to(device)
        # Categogrical encoding
        labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
        logits, reconstructions = model(images)
        pred_labels = torch.argmax(logits, dim=1)
        correct += torch.sum(pred_labels == torch.argmax(labels, dim=1)).item()
        total += len(labels)
    print('Accuracy: {}'.format(correct / total))

    # Save model
    torch.save(model.state_dict(), './model/capsnet_ep{}_acc{}.pt'.format(EPOCHES, correct / total))


if __name__ == '__main__':
    train, val, test = datasets_control.get_datasets()
    train = get_data_loader_A_APTOS(train, 28)
    valid = get_data_loader_A_APTOS(val, 28)
    main(train, valid)
