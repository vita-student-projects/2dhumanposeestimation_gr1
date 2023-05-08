#############################################################
# Script to check whether OpenPifPaf is correctly installed #
#############################################################

# Take a CrowdPose image and run a prediction, store the example output
python -m openpifpaf.predict data-crowdpose/images/110000.jpg --image-output out/sanity-check-image.jpg --json-output out/sanity-check-annotation.json
