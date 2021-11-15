import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from facenet_pytorch import MTCNN
from model import Net
import CONFIG
import warnings
warnings.filterwarnings('ignore')
image_size = CONFIG.IMAGE_SIZE
use_gpu = torch.cuda.is_available()

# create the detector, using default weights

mtcnn = MTCNN(keep_all=True, device='cuda') if use_gpu else MTCNN(keep_all=True)

def inference_video(video_path, checkpoint_path, classes, webcam=False):
    model = Net(num_classes=CONFIG.NUM_CLASSES)

    if use_gpu:
        model = model.cuda()
    transform = transforms.Compose([transforms.Resize(image_size), transforms.PILToTensor()])
    
    assert os.path.exists(CONFIG.CHECKPOINT_PATH), "checkpoint not found"
    print("checkpoint to resume: ", checkpoint_path)
    tmp = torch.load(checkpoint_path)
    model.load_state_dict(tmp['state'])
    print("checkpoint restored !!!")

    cap = cv2.VideoCapture(0) if webcam else cv2.VideoCapture(video_path)

    while True:
        ret, frame_orig = cap.read()

        boxes, probs = mtcnn.detect(Image.fromarray(frame_orig))

        if boxes is None:
            continue
        for i in range(len(boxes)):
            box = boxes[i]
            prob = probs[i]

            if prob <= 0.8:
                continue

            x1, y1, x2, y2 = [int(cor) for cor in box]
            face = frame_orig[y1:y2, x1:x2]

            face = transform(Image.fromarray(face))
            face = face / 255.

            if use_gpu:
                face = face.cuda()
            
            face = face.view(1, 3, image_size[0], image_size[1])
            with torch.no_grad():
                logps = model(face)
            
            # post process 
            ps = torch.exp(logps)
            probab = list(ps.cpu()[0])
            predicted_label = probab.index(max(probab))
            class_ = f"{classes[predicted_label].replace(classes[predicted_label][-1], '')}"

            frame_orig = cv2.rectangle(frame_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame_orig = cv2.putText(frame_orig, class_, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("predictions", frame_orig)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    with open(CONFIG.PATH_TO_CLASSES_TXT, 'r') as f:
        classes = f.readlines()
    
    inference_video(CONFIG.VIDEO_PATH, CONFIG.CHECKPOINT_PATH, classes, webcam=False)
