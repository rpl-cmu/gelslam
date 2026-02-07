#!/bin/bash
# Script to download sample data for GelSLAM

DOWNLOAD_URL="https://github.com/rpl-cmu/gelslam/releases/download/v1.0.0-data/sample_data.zip"

echo "Downloading sample data..."
mkdir -p data
wget -O data/sample_data.zip "$DOWNLOAD_URL"

if [ $? -eq 0 ]; then
    echo "Download complete. Unzipping..."
    unzip -o data/sample_data.zip -d data/
    echo "Done! Data is ready in the 'data' directory."
else
    echo "Error: Download failed. Please check your internet connection or the URL."
    exit 1
fi
