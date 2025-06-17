import pandas as pd
import shutil
import argparse
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training data preprocessor")
    parser.add_argument("inpath", type=str, help="path to the INPUT image folder")
    parser.add_argument("outpath", type=str, help="path to the OUTPUT image folder")
    parser.add_argument("attr_list", type=str, help="path to the ATTRIBUTE LIST")
    parser.add_argument("attr_labels", type=str, help="path to the ATTRIBUTE LIST")
    parser.add_argument(
        "--neg_class", 
        type=str, default="Eyeglasses", help="undesired attribute class (see file attr_names.txt)"
    )
    args = parser.parse_args()

    attributes = list(np.loadtxt(args.attr_list, dtype=str))
    attributes.insert(0, "ID")

    attribute_label_df = pd.read_csv(args.attr_labels, delim_whitespace=True, dtype='object', header=None)
    attribute_label_df.columns = attributes
    print(attribute_label_df.head(3))
    
    image_folder = args.inpath
    dst_folder = args.outpath
    neg_attr = args.neg_class
    os.makedirs(f"{dst_folder}/removed_{neg_attr}/data", exist_ok=True)

    for idx, img_row in attribute_label_df.iterrows():
        img_id, img_neg_attr_label = str(img_row["ID"]), int(img_row[neg_attr])
        if img_neg_attr_label == -1:
            shutil.copy(f"{image_folder}/{img_id}", f"{dst_folder}/removed_{neg_attr}/data/{img_id}")
