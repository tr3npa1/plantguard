from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "val"
TEST_DIR = PROJECT_ROOT / "data" / "test"
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2

class PlantDiseaseDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = Path(root_dir)   #root dir will be train
        self.transform = transform
        self.image_extensions = {".jpg",".jpeg",".png",".bmp"}
        self.class_names = sorted([
            folder.name
            for folder in self.root_dir.iterdir()
            if folder.is_dir()
        ])
        self.class_to_idx = {
            class_name: idx
            for idx,class_name in enumerate(self.class_names)
        }
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            label = self.class_to_idx[class_name]
            for image_path in class_dir.iterdir():
                if(
                    image_path.is_file() 
                    and image_path.suffix.lower() in self.image_extensions
                ):
                    self.samples.append((image_path,label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image,label
    
def get_transform():
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees = 15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485,0.456, 0.406],
            std = [0.229,0.224,0.225]
        )
    ])
    return train_transforms,eval_transforms

def get_datasets():
    train_transforms,eval_transforms = get_transform()
    train_dataset = PlantDiseaseDataset(
        root_dir = TRAIN_DIR,
        transform = train_transforms
    )
    val_dataset = PlantDiseaseDataset(
        root_dir = VAL_DIR,
        transform = eval_transforms
    )
    test_dataset = PlantDiseaseDataset(
        root_dir = TEST_DIR,
        transform = eval_transforms
    )
    return train_dataset,val_dataset,test_dataset

def get_dataloaders(batch_size = BATCH_SIZE, num_workers = NUM_WORKERS):
    train_dataset,val_dataset,test_dataset=get_datasets()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers=num_workers
    )
    class_names = train_dataset.class_names
    return train_loader,val_loader,test_loader,class_names

if __name__=="__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"Number of classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"First 5 labels: {labels[:5]}")

