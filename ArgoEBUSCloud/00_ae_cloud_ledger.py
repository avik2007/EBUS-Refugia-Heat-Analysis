"""
=============================================================================
THE CLOUD LEDGER
=============================================================================
A quick CLI tool to inventory processed datasets sitting in your AWS S3 
Data Lake without having to log into the AWS Web Console.
=============================================================================
"""

import s3fs
from ebus_core.ae_utils import get_ebus_registry

def list_s3_datasets(region="california"):
    """Scans your AWS S3 bucket and prints a clean ledger of all processed data."""
    registry = get_ebus_registry()
    
    if region not in registry:
        print(f"❌ Region '{region}' not found in ae_utils registry.")
        return

    bucket = registry[region]['s3_bucket']
    print(f"\n🔍 Scanning AWS S3 Bucket: [ {bucket} ] for {region.upper()}...")
    print("-" * 70)

    # Initialize the S3 file system using your local AWS credentials
    fs = s3fs.S3FileSystem(anon=False)

    try:
        # List all contents in the bucket
        contents = fs.ls(bucket)
        
        # Filter for the .parquet datasets we generated
        parquet_files = [path for path in contents if path.endswith('.parquet')]

        if not parquet_files:
            print(f"📭 No datasets found. The bucket is currently empty.")
        else:
            print(f"✅ Found {len(parquet_files)} ready-to-use dataset(s):\n")
            for i, file_path in enumerate(parquet_files, 1):
                # Strip out the bucket name to just show the clean file name
                file_name = file_path.split('/')[-1]
                print(f"   {i}. 💾 {file_name}")
                
    except FileNotFoundError:
        print(f"⚠️  Bucket '{bucket}' does not exist yet. Run your ingestion script first!")
    except Exception as e:
        print(f"❌ AWS Connection Error: {e}")
        
    print("-" * 70)
    print("💡 TIP: Copy the filename above to use in your plotting scripts.\n")

if __name__ == "__main__":
    # You can easily swap this to "humboldt", "canary", etc., to check other buckets
    list_s3_datasets("california")