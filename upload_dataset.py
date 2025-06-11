from mistralai import Mistral
import os

#api_key = os.environ["MISTRAL_API_KEY"]
api_key = "l7ZmesIQ6QsjfmRwTGjxDku2k5WLR8Sz"
filename = "instruction_dataset.jsonl"

client = Mistral(api_key=api_key)

ultrachat_chunk_train = client.files.upload(file={
    "file_name": filename,
    "content": open(filename, "rb"),
})