from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import os
from tqdm import tqdm
import utils
import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def collate_fn(data):
    """Transform a list of tuples into a batch

    Args:
        data (list[tuple[np.ndarray, int, np.ndarray, np.ndarray]]): a list of tuples sampled from a dataset

    Returns:
        tensor: a list of tensor data
    """
    # data = a list of tuples
    batch_size = len(data)
    batch1 = [data[i][0] for i in range(batch_size)]
    batch2 = [data[i][1] for i in range(batch_size)]
    batch3 = [data[i][2] for i in range(batch_size)]
    return batch1, batch2, batch3


class ImageData(Dataset):
    def __init__(self, img_folder, meta_data):
        """Initialize the dataset

        Args:
            img_folder (str): folder containing images
            meta_data (pd.DataFrame): metadata of the dataset
        """
        self.meta_data = meta_data
        self.img_folder = img_folder


    def __len__(self):
        return len(self.meta_data)
    
    def __getitem__(self, idx):
        """Return an image, its label, and its name

        Args:
            idx (int): index of the image

        Returns:
            (PIL.Image.Image, int, str): an image, its label, and its name
        """
     
        image_path = self.meta_data.iloc[idx]["img_filename"]
        temp = Image.open(os.path.join(self.img_folder,image_path))
        if temp.mode != "RGB":
            temp = temp.convert(mode="RGB")
        image = temp.copy()
        temp.close()
        image_name = os.path.split(image_path)[-1]
        label = self.meta_data.iloc[idx]["y"]
        return image, label, image_name

class VITGPT2_CAPTIONING:
    def __init__(self, max_length=16, num_beams=4):
        """Initialize the ViT-GPT2 model for image captioning

        Args:
            max_length (int, optional): Maximum description length. Defaults to 16.
            num_beams (int, optional): Number of beams used in beam search. Defaults to 4.
        """
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        gpu = ",".join([str(i) for i in utils.get_free_gpu()[0 : 1]])
        utils.set_gpu(gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        self.model.eval()

        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


    def predict_step(self, data, names):
        """Generate image captions for a batch of images

        Args:
            data (list[PIL.Image.Image]): a list of Image objects
            names (list[str]): a list of image names

        Returns:
            list[str]: a list of image captions in the format of "image_name,caption"
        """
        with torch.no_grad():
            pixel_values = self.feature_extractor(images=data, return_tensors="pt").pixel_values
            pixel_values = pixel_values.cuda()
            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            msgs = []
            for i in range(len(preds)):
                msgs.append(f"{names[i]},{preds[i].strip()}")
        return msgs

    def get_img_captions(self, img_folder, csv_path, batch_size=256):
        """Generate image captions for images specified in the csv file

        Args:
            img_folder (str): folder containing images
            csv_path (str): metadata csv file
            batch_size (int, optional): Batch size. Defaults to 256.

        Raises:
            ValueError: if the csv file does not exist

        Returns:
            str: path to the generated csv file containing image captions
        """
        if not os.path.exists(csv_path):
            raise ValueError(f"{csv_path} does not exist")
        metadata_df = pd.read_csv(csv_path)
        metadata_df = metadata_df[metadata_df["split"] != 2]

        save_path = os.path.join(img_folder, f"vit-gpt2_captions.csv")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                caption_lines = f.readlines()
            if len(caption_lines) == len(metadata_df):
                print(f"{save_path} have been generated")
                return save_path

        dataset = ImageData(img_folder, metadata_df)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,collate_fn=collate_fn)
        count = 0
        timer = utils.Timer()
        with open(save_path, "w") as fout:
            for data, labels, names in dataloader:
                msgs = self.predict_step(data, names)
                write_info = '\n'.join([f"{msgs[i]},{labels[i]}" for i in range(len(msgs))])
                fout.write(f"{write_info}\n")
                fout.flush()
                count += batch_size
                elapsed_time = timer.t()
                est_time = elapsed_time / count * len(metadata_df)
                print(f"Progress: {count / len(metadata_df) * 100:.2f}% {utils.time_str(elapsed_time)}/est:{utils.time_str(est_time)}   ", end="\r")
        return save_path


class BLIP_CAPTIONING:
    def __init__(self, max_length=16, num_beams=4):
        """Initliaze the BLIP model for image captioning

        Args:
            max_length (int, optional): Maximum description length. Defaults to 16.
            num_beams (int, optional): Number of beams used in beam search. Defaults to 4.
        """
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
    
        gpu = ",".join([str(i) for i in utils.get_free_gpu()[0:1]])
        utils.set_gpu(gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.model.eval()
        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def predict_step(self, data, names):
        """Generate image captions for a batch of images

        Args:
            data (list[PIL.Image.Image]): a list of Image objects
            names (list[str]): a list of image names

        Returns:
            list[str]: a list of image captions in the format of "image_name,caption"
        """
        with torch.no_grad():
            text = "there"
            inputs = self.processor(data, [text]*len(data), return_tensors="pt").to("cuda")
            output_ids = self.model.generate(**inputs)
            preds = self.processor.batch_decode(output_ids, skip_special_tokens=True)

            msgs = []
            for i in range(len(preds)):
                msgs.append(f"{names[i]},{preds[i].strip()}")
        return msgs

    def get_img_captions(self, img_folder, csv_path, batch_size=256):
        """Generate image captions for images specified in the csv file

        Args:
            img_folder (str): folder containing images
            csv_path (str): metadata csv file
            batch_size (int, optional): Batch size. Defaults to 256.

        Raises:
            ValueError: if the csv file does not exist

        Returns:
            str: path to the generated csv file containing image captions
        """
        if not os.path.exists(csv_path):
            raise ValueError(f"{csv_path} does not exist")
        
        metadata_df = pd.read_csv(csv_path)
        metadata_df = metadata_df[metadata_df["split"] != 2]
      

        save_path = os.path.join(img_folder, "blip_captions.csv")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                caption_lines = f.readlines()
            if len(caption_lines) == len(metadata_df):
                print(f"{save_path} has been generated")
                return
        dataset = ImageData(img_folder, metadata_df)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,collate_fn=collate_fn)
        timer = utils.Timer()
        count = 0
        with open(save_path, "w") as fout:
            for data, labels, names in dataloader:
                msgs = self.predict_step(data, names)
                write_info = '\n'.join([f"{msgs[i]},{labels[i]}" for i in range(len(msgs))])
                fout.write(f"{write_info}\n")
                fout.flush()
                count += batch_size
                elapsed_time = timer.t()
                est_time = elapsed_time / count * len(metadata_df)
                print(f"Progress: {count / len(metadata_df) * 100:.2f}% {utils.time_str(elapsed_time)}/est:{utils.time_str(est_time)}   ", end="\r")    
        return save_path
