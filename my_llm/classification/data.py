import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd

def download_and_prepare_data(url: str = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip") -> None:
    """
    Download, extract, and process the SMS Spam Collection dataset.
    """
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download.")
    else:
        try:
            with urllib.request.urlopen(url) as response:
                with open(zip_path, "wb") as out_file:
                    out_file.write(response.read())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extracted_path)

            original_file_path = Path(extracted_path) / "SMSSpamCollection"
            os.rename(original_file_path, data_file_path)
            print(f"Dataset saved at {data_file_path}.")

        except Exception as e:
            print(f"Download failed: {e}")
            raise

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    num_spam = df[df.Label == "spam"].shape[0]
    ham_subset = df[df.Label == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df.Label == "spam"]])
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("validation.csv", index=False)
    test_df.to_csv("test.csv", index=False)

def random_split(df: pd.DataFrame, train_frac: float, validation_frac: float):
    """
    Randomly split a dataframe into train/validation/test sets.
    """
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:val_end], df[val_end:]