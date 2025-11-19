import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class CelebADataset(Dataset):
    """
    CelebA dataset loader that handles images, attributes, bounding boxes, and landmarks.
    
    Args:
        img_dir (str): Path to the directory containing images
        attr_path (str): Path to the attributes CSV file
        bbox_path (str): Path to the bounding boxes CSV file
        landmark_path (str): Path to the landmarks CSV file
        partition_path (str): Path to the partition CSV file
        split (str): One of 'train', 'val', or 'test'. Default: 'train'
        transform (callable, optional): Optional transform to apply to images
    """
    
    SPLIT_MAP = {"train": 0, "val": 1, "test": 2}
    TARGET_SIZE = (128, 128)
    
    def __init__(self, img_dir, attr_path, bbox_path, landmark_path, partition_path, 
                 split='train', transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Read and prepare attributes
        self.attrs = self._load_attributes(attr_path)
        
        # Read and prepare bounding boxes
        bbox = self._load_dataframe(bbox_path)
        
        # Read and prepare landmarks
        landmarks = self._load_dataframe(landmark_path)
        
        # Read partition file
        parts = pd.read_csv(partition_path, sep=r"\s+", header=None, 
                           names=['image_id', 'split'])
        parts["image_id"] = parts["image_id"].str.strip()
        
        # Merge all tables
        df = parts.merge(self.attrs, left_on='image_id', right_index=True)
        df = df.merge(bbox, left_on='image_id', right_index=True, 
                     suffixes=('', '_bbox'))
        df = df.merge(landmarks, left_on='image_id', right_index=True, 
                     suffixes=('', '_lm'))
        
        # Filter by split
        df = df[df["split"] == self.SPLIT_MAP[split]]
        
        self.image_names = df["image_id"].values
        
        # Store attributes (40 columns)
        attr_cols = list(self.attrs.columns)
        self.attrs_data = ((df[attr_cols].values + 1) // 2).astype("float32")
        
        # Store bounding boxes
        bbox_cols = ["x_1", "y_1", "width", "height"]
        self.bbox = df[bbox_cols].values.astype("float32")
        
        # Store landmarks (10 values)
        lm_cols = ["lefteye_x", "lefteye_y", "righteye_x", "righteye_y",
                   "nose_x", "nose_y", "leftmouth_x", "leftmouth_y", 
                   "rightmouth_x", "rightmouth_y"]
        self.landmarks = df[lm_cols].values.astype("float32")
    
    @staticmethod
    def _load_dataframe(path):
        """Load and standardize index names for a dataframe."""
        df = pd.read_csv(path, sep=r"\s+", skiprows=1)
        df.index = df.index.astype(str).str.strip()
        
        if not df.index[0].endswith(".jpg"):
            df.index = df.index + ".jpg"
        
        return df
    
    def _load_attributes(self, attr_path):
        """Load attributes from CSV file."""
        attrs = pd.read_csv(attr_path, sep=r"\s+", skiprows=1)
        attrs.index = attrs.index.astype(str).str.strip()
        
        if not attrs.index[0].endswith(".jpg"):
            attrs.index = attrs.index + ".jpg"
        
        return attrs
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_names)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            dict: Contains 'image', 'attributes', 'bbox', and 'landmarks'
        """
        # Load image
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Get original dimensions
        orig_w, orig_h = image.size  # (178, 218)
        
        # Copy bbox and landmarks to avoid modifying originals
        bbox = self.bbox[idx].copy()
        landmarks = self.landmarks[idx].copy()
        
        # Apply image transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Calculate scaling factors for target size
        new_w, new_h = self.TARGET_SIZE
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        # Scale bounding box
        bbox[0] *= scale_x
        bbox[1] *= scale_y
        bbox[2] *= scale_x
        bbox[3] *= scale_y
        
        # Scale landmarks
        landmarks[0::2] *= scale_x
        landmarks[1::2] *= scale_y
        
        # Convert to tensors
        return {
            'image': image,
            'attributes': torch.tensor(self.attrs_data[idx], dtype=torch.float32),
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'landmarks': torch.tensor(landmarks, dtype=torch.float32)
        }
        