import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np
import argparse

def get_v(path):
    return float( int(path[3:6]) ) / 100.0

def get_w(path):
    return  float(int(path[7:11]) - 3000  ) / 1000.0

class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path)
        v = float(get_v(os.path.basename(image_path)))
        w = float(get_w(os.path.basename(image_path)))       
        # if float(np.random.rand(1)) > 0.5:
        #     image = transforms.functional.hflip(image)
        #     x = -x
        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor([v, w]).float()


def train_pilot():
    dataset = XYDataset('dataset_vw', random_hflips=False)
    test_percent = 0.1
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=4)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 2)
    device = torch.device('cuda')
    model = model.to(device)
    NUM_EPOCHS = 70
    BEST_MODEL_PATH = 'best_steering_model_xy.pth'
    best_loss = 1e9
    optimizer = optim.Adam(model.parameters())
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            train_loss += float(loss)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
    
        model.eval()
        test_loss = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            test_loss += float(loss)
        test_loss /= len(test_loader)
        
        print('%f, %f' % (train_loss, test_loss))
        if test_loss < best_loss:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_loss

def train_collision():
    print('tbd')

def parse_args():
	# Parse input arguments
	desc = 'model train'
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--width', dest='image_width',
                        help='image width [320]',
                        default=320, type=int)
	parser.add_argument('--height', dest='image_height',
                        help='image height [240]',
                        default=240, type=int)
	parser.add_argument('--model', '-m', dest='model', help='model to train[collision,  pilot, object]',  default='collision')
	args = parser.parse_args( )
	return args

def main():
    train_pilot()
	# args = parse_args()
	# print('Called with args:')
	# print(args)
    # if args.model == 'pilot':
    #     train_pilot()
    # elif args.model == 'collision':
    #     train_collision()

if __name__ == '__main__':
	main()