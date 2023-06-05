import os
import sys
import openpifpaf.video


def predict_video(video):
    print(video)
    video_name = os.path.splitext(video)[0]
    args = f"--checkpoint checkpoints/shufflenet_pruned2.pt --json-output test-videos/{video_name}.json --source test-videos/{video}"
    print(args)
    sys.argv.extend(args.split())
    openpifpaf.video.main()

def main():
    files = os.listdir("./test-videos/")
    videos = [video for video in files if os.path.splitext(video)[1] == ".mp4"]
    for video in videos:
        predict_video(video)

if __name__ == "__main__":
    main()
