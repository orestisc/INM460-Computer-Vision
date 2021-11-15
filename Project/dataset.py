from PIL import Image
import os



class DatasetLoader:
    def __init__(self, path, transforms, mode="train"):
        
        self.mode = mode
        print('data_file:', path)
        path_to_images = os.path.join(path, "train") if self.mode == "train" else os.path.join(path, "test")
        labels_txt_path = os.path.join(path, "labels", "list_label_train.txt") if self.mode == 'train' else os.path.join(path, "labels", "list_label_test.txt")

        self.meta = {
            "images": [],
            "labels": []
        }
        with open(labels_txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    image_name, label = line.split()
                    self.meta["images"].append(os.path.join(path_to_images, image_name))
                    self.meta["labels"].append(int(label)-1)
                except Exception as e:
                    print(f"ERROR -> {str(e)}") # skip that line
            print(f"total {self.mode}ing images:", len(self.meta["images"]))

        self.transform = transforms[self.mode]

    def __getitem__(self, i):
        image_path = self.meta['images'][i]
        img = Image.open(image_path).convert('RGB')

        img = self.transform(img)
        img = img / 255.
        target = self.meta['labels'][i]

        return img, target

    def __len__(self):
        return len(self.meta['images'])
