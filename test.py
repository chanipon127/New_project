import zipfile

zip_path = r"C:\Users\User\Documents\New_project\all_dataset\S1\2final_model_bi.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    print(z.namelist()[:20])


import zipfile

zip_path = r"C:\Users\User\Documents\New_project\all_dataset\S1\2final_model_bi.zip"
extract_path = r"C:\Users\User\Documents\New_project\all_dataset\S1\2final_model_bi"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted")


import os

s3_path = r"C:\Users\User\Documents\New_project\all_dataset\S3"

for root, dirs, files in os.walk(s3_path):
    for file in files:
        print(os.path.join(root, file))


