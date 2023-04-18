###########################################
# Script to install the CrowdPose dataset #
###########################################

# Change to the data directory
cd crowdpose

# Download the images
gdown https://drive.google.com/uc?id=1VprytECcLtU4tKP32SYi_7oDRbw7yUTL
# Unzip images
unzip images.zip
# Remove the zip file
rm images.zip

# Download the annotations
gdown https://drive.google.com/drive/folders/1Ch1Cobe-6byB7sLhy8XRzOGCGTW2ssFv --folder
# Rename the downloaded folder
mv /CrowdPose /annotations

# Change back to root directory
cd ..