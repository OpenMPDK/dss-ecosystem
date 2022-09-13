# USAGE
# python3 object_detector_predict.py --input dataset/images/face/image_0131.jpg
# Or
# python3 object_detector_predict.py --input dataset/test_images/test_image_list.txt

from torchvision import transforms
import mimetypes
import argparse
import imutils
import pickle
import torch
import cv2
import os
from utils.config import Config


# construct the argument parser and parse the arguments
def ArgumentParser():
    parser = argparse.ArgumentParser(description='Benchmarking tool')
    parser.add_argument("--config", "-cfg", type=str, required=False, help='Specify configuration file path')
    parser.add_argument("--computation", "-comp", type=str, required=False, default="CPU",
                        help='CPU/GPU based computation.')
    parser.add_argument("--input", "-i", required=True,
                        help="path to input image/text file of image paths")

    options = vars(parser.parse_args())
    return options


params = ArgumentParser()
config_obj = Config(params)
config = config_obj.get_config()

device = params["computation"].lower() if params["computation"] else config["device"].lower()  # GPU /CPU
if not device or device == "gpu":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

categories = config["dataset"][config["dataset"]["choice"]]["label"]
config_mean = config["dataset"][config["dataset"]["choice"]]["mean"]
config_std = config["dataset"][config["dataset"]["choice"]]["std"]

# determine the input file type, but assume that we're working with single input image
filetype = mimetypes.guess_type(params["input"])[0]
imagePaths = [params["input"]]

# if the file type is a text file, then we need to process multiple images with
# each line of the txt file containing individual test image **absolute** paths
try:
    if "text/plain" == filetype:
        # load the image paths in our testing file
        with open(params["input"]) as file:
            imagePaths = file.read()
            imagePaths = imagePaths.strip().split("\n")
except FileNotFoundError:
    print('[ERROR] No file found with the given name. Please check whether the path or name of the file is correct!')

print(f"[INFO] Image file(s) read: {imagePaths}")

# load our object detector (setting it to evaluation mode) and label encoder pickle from disk
print("[INFO] loading object detector...")
model_path = os.path.sep.join(
    [config["storage"][config["storage"]["format"]][config["storage"][config["storage"]["format"]]["choice"]]["base_output_dir"],
     config["storage"][config["storage"]["format"]][config["storage"][config["storage"]["format"]]["choice"]]["saved_model_name"]])
model = torch.load(model_path).to(device)
model.eval()
print("[INFO] Model loaded and set to `eval` mode...")

# define normalization transforms
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=config_mean, std=config_std)
])

# loop over the images that we'll be testing using our model
for imagePath in imagePaths:
    # load the image, copy it, swap its colors channels, resize it, and
    # bring its channel dimension forward
    image = cv2.imread(imagePath)
    fileName = imagePath.strip().split('.')[0].split('/')[-1]
    copied_img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))

    # convert image to PyTorch tensor, normalize it, flash it to the
    # current device, and add a batch dimension
    image = torch.from_numpy(image)
    image = transforms(image).to(device)
    image = image.unsqueeze(0)
    print("[INFO] Image preprocessed and going for predictions...")

    # predict the bounding box of the object along with the class label
    (boxPreds, labelPreds) = model(image)
    (startX, startY, endX, endY) = boxPreds[0]

    # determine the class label with the largest predicted probability
    labelPreds = torch.nn.Softmax(dim=-1)(labelPreds)
    i = labelPreds.argmax(dim=-1).cpu()
    label = categories[i]

    # resize the original image such that it fits on our screen, and
    # grab its dimensions
    copied_img = imutils.resize(copied_img, width=600)
    (h, w) = copied_img.shape[:2]

    # scale the predicted bounding box coordinates based on the image dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    # draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(copied_img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2)
    cv2.rectangle(copied_img, (startX, startY), (endX, endY),
                  (0, 255, 0), 2)
    print("[INFO] Label & bboxes predicted and written to output image...")

    cv2.imwrite(os.path.sep.join(
        [config["storage"][config["storage"]["format"]][config["storage"][config["storage"]["format"]]["choice"]]["predictions_path"],
         fileName + '.jpg']), copied_img)

    saved_img = os.path.abspath(os.path.sep.join(
        [config["storage"][config["storage"]["format"]][config["storage"][config["storage"]["format"]]["choice"]]["predictions_path"],
         fileName + ".jpg"]))
    print(f'[INFO] Output image written to: {saved_img}')
