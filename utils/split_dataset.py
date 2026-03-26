import os
import random
import shutil

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    random.seed(42)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        for split_name, split_images in zip(
            ["train", "val", "test"],
            [train_images, val_images, test_images]
        ):
            split_path = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_path, exist_ok=True)

            for img in split_images:
                shutil.copy(
                    os.path.join(class_path, img),
                    os.path.join(split_path, img)
                )

    print("Dataset successfully split into train/val/test!")
