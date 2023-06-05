###########################################
# Script to install the CrowdPose dataset #
###########################################

# Download the json format 
gdown --id 1iqC4g0YVN-N9WkRDPXMXM9iqJK6nGLcz
# Download the video folders
gdown --id 1F7rvvHeFF3ZVyl3uDFFRKRTlbHSzsQxn --folder
gdown --id 1Cnh2HVxwX-nTdV_J4tiGvZPojs40RCrx --folder
gdown --id 1pACmQLnojWcNH9gkbY3xwHDzurFqMSyp --folder
gdown --id 1rYuKNNEtZ2hJ9j4YZix4wSv05XByvDDi --folder
gdown --id 1UgRs_UcdDUdyFHO5Q2sqqzRrt30BQAbb --folder

# Move to to current directory
cp -a mp4/. .
rm -rf mp4/
