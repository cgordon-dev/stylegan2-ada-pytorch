import boto3
import os

bucket = 'aws-gpu-monitoring-logs'
prefix = 'stylegan2-ada-results'
tag_key = 'gpu-tag'
tag_value = 'nvidia-t4'
local_dir = './training-runs'

s3 = boto3.client('s3')

def upload_dir(path, bucket, prefix):
    for root, _, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, path)
            s3_path = f"{prefix}/{relative_path}"
            print(f"Uploading {full_path} to s3://{bucket}/{s3_path}")
            s3.upload_file(full_path, bucket, s3_path,
                           ExtraArgs={'Tagging': f'{tag_key}={tag_value}'})

upload_dir(local_dir, bucket, prefix)