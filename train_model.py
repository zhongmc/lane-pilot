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
import cv2

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
    model = models.resnet18(pretrained=False)
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
    dataset = datasets.ImageFolder(
        'dataset',
        transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    print( dataset.classes )
    print( dataset.classes_to_idx  )

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )   

    print('downloading  alexnet-owt-4df8aa71.pth .... ')
    model = models.alexnet(pretrained=True)     
    # will download https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth to /home/zhongmc/.cache/torch/checkpoints/alexnet-owt-4df8aa71.pth
    # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)
    #he alexnet model was originally trained for a dataset that had 1000 class labels,
    print('done.')
    device = torch.device('cuda')
    model = model.to(device)

    NUM_EPOCHS = 30
    BEST_MODEL_PATH = 'best_model.pth'
    best_accuracy = 0.0

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        
        test_error_count = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        
        test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
        print('%d[%d]: %f' % (epoch, NUM_EPOCHS, test_accuracy))
        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = test_accuracy
    print("Done " )

def test_pilot(file):
    pass


def test_collision(imgfile):
    print("init ai collision detecter...")
    dataset = datasets.ImageFolder(
        'dataset',
        transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    print( dataset.classes )
    print( dataset.class_to_idx  )

    model = torchvision.models.alexnet(pretrained=False)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)
    model.load_state_dict(torch.load('best_model.pth'))
      #  IncompatibleKeys(missing_keys=[], unexpected_keys=[])
    device = torch.device('cuda')
    model = model.to(device)
    mean = 255.0 * np.array([0.485, 0.456, 0.406])
    stdev = 255.0 * np.array([0.229, 0.224, 0.225])
    normalize = torchvision.transforms.Normalize(mean, stdev)
    print('done!')
    if imgfile is not None:
        print('stop:', imgfile)
        image=cv2.imread(imgfile )
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        x = normalize(x)
        x = x.to(device)
        x = x[None, ...]
        y = model(x)
        prtf_result(y)
 
    imgfile = 'dataset/blocked/06-14-164407.jpg'
    print('ojs:', imgfile)
    image=cv2.imread(imgfile )
    x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    y = model(x)
    prtf_result(y)
 
    imgfile = 'dataset/blocked/06-14-161132.jpg'
    print('ojs:', imgfile)
    image=cv2.imread(imgfile )
    x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    y = model(x)
    prtf_result(y)

    imgfile = 'dataset/free/06-14-164514.jpg'
    print('free:', imgfile)
    image=cv2.imread(imgfile )
    x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    y = model(x)
    prtf_result( y )
    # idx = np.argmax( y.flatten())
    # print( idx )
    imgfile = 'cap_imgs/img06-01-205720.jpg'
    print('stop:', imgfile)
    image=cv2.imread(imgfile )
    x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    y = model(x)
    prtf_result( y )


def prtf_result( y ):
    print( y )
    y = F.softmax(y, dim=1)
    print(y)
    y = y.flatten()
    ny = y.detach().cpu()
    ny = ny.numpy()
    idx = np.argmax( ny )
    print( idx, ny[idx])
    print('.........')



def parse_args():
	# Parse input arguments
    desc = 'model train'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--width', dest='image_width',help='image width [320]',default=320, type=int)    
    parser.add_argument('--height', dest='image_height',help='image height [240]',default=240, type=int)
    parser.add_argument('--model', '-m', dest='model', help='model to train[collision,  pilot, object]',  default='collision')
    parser.add_argument('--file', '-f', dest='file', help='img file to test ')
    parser.add_argument('--train', dest='train', help='do train or test',  action='store_true')
    args = parser.parse_args( )
    return args

def main():
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.train :
        if args.model == 'pilot':
            train_pilot()
        elif args.model == 'collision':
            train_collision()

    else:
        if args.model == 'pilot':
            test_pilot(args.file)
        elif args.model == 'collision':
            test_collision(args.file )
        return

if __name__ == '__main__':
	main()
