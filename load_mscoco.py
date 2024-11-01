import json
from torch.utils.data import Dataset
from PIL import Image


#   Adopted and adapted from:  https://github.com/clemneo/llava-interp/blob/main/src/ImageDatasets.py
class COCOImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, filter_fn=None):
        self.data_dir = data_dir
        # self.data_type = data_type
        # self.coco = COCO(ann_file)
        self.transform = transform

        # self.filter_fn = filter_fn
        # self.filtered_img_ids = self._filter_images()

        self.img_list = self._get_img_data()

    def _get_img_data(self):
        img_list = []

        # Open and read the JSON file
        with open(f'{self.data_dir}/annotations/captions_val2017.json', 'r') as file:
            data = json.load(file)

        for img in data['images']:
            img_data = {}
            # img_data['id'] = img['id'] 
            img_data['file_name'] = img['file_name']
            
            for ann in data['annotations']:
                if ann['image_id'] == img['id']:
                    img_data['caption'] = ann['caption']
                    break
            
            img_list.append(img_data)

        return img_list
        
    # def _filter_images(self):
    #     filtered_ids = []

    #     if self.filter_fn is None:
    #         filtered_ids = self.coco.getImgIds()
    #         print("No filter passed! All images will be in the dataset.")
    #         return filtered_ids

    #     for img_id in self.coco.getImgIds():
    #         ann_ids = self.coco.getAnnIds(imgIds=img_id)
    #         anns = self.coco.loadAnns(ann_ids)

    #         if self.filter_fn(anns):
    #             filtered_ids.append(img_id)

    #     print(f"{len(filtered_ids)}/{len(self.coco.getImgIds())} images passed the filter.")

    #     return filtered_ids
    
    def __len__(self):
        return len(self.img_list)
         
    def __getitem__(self, idx: int):
        img_data = self.img_list[idx]
        img = Image.open(f"{self.data_dir}/val2017/spoof_class/{img_data['file_name']}")
        capt = img_data['caption']

        if self.transform:
            img = self.transform(img)        

        return img, capt