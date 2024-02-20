from torch.utils.data import Dataset
import os


class GenerativeAIDataset(Dataset):

    def __init__(self, folder, transform=None, class_c=None) -> None:
        all_classes = [
            d for d in os.listdir(folder)\
                if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        
        data = []

        for c in all_classes:
            d = os.path.join(folder, c)
            if class_c is not None and c != class_c:
                continue
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append(path)

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        data = {'path': path}

        if self.transform is not None:
            data = self.transform(data)

        return data
    