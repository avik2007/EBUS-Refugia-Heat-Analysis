import s3fs
import os

fs = s3fs.S3FileSystem(anon=False)
bucket = "argo-ebus-project-data-abm"

old_path = f"{bucket}/california_20150101_20151231_res1_0x1_0.parquet"
new_path = f"{bucket}/california_testbox_20150101_20151231_res1_0x1_0.parquet"
temp_local = "temp_testbox.parquet"

try:
    print(f"📥 Downloading {old_path} to local...")
    fs.get(old_path, temp_local)
    
    if os.path.exists(temp_local):
        print(f"📤 Uploading to {new_path}...")
        fs.put(temp_local, new_path)
        
        # Verify the upload worked before deleting
        if fs.exists(new_path):
            print("🗑️ Deleting old file from S3...")
            fs.rm(old_path)
            print("✅ SUCCESS: File successfully bridged and renamed.")
        else:
            print("❌ ERROR: Upload reported success but file is missing in S3.")
    else:
        print("❌ ERROR: Local download failed.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")

finally:
    if os.path.exists(temp_local):
        os.remove(temp_local) 