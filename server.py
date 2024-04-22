from fastsam import FastSAM, FastSAMPrompt
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np


class Request(BaseModel):
    base64: str
    link: str

class Response(BaseModel):
    text: str


class Model:
    # FastSAM x model: 138MB
    STATIC_MODEL_X = FastSAM('FastSAM-x.pt')

    def __init__(self):
        self.__dump_temp = 'dump.png'
        

    def infer(self, link):
        image = cv2.imread(link)
      
        height = image.shape[0]
        width = image.shape[1]
      
        # TODO: might need to save the image locally as 
        # later the base64 might be received instead of the image link.
        model = Model.STATIC_MODEL_X(link,
                                                  device='cpu',
                                                  retina_masks=True,
                                                  imgsz=1024,
                                                  conf=0.4,
                                                  iou=0.9)
        prompt_process = FastSAMPrompt(link,
                                       model,
                                       device='cpu')


        print("LOG - calculate annotation")
        ann = prompt_process.point_prompt(points=[[int(width/2),int(height/2)]], pointlabel=[1])

        binary_mask = np.where(ann > 0.5, 1, 0)


        print("LOG - generate new image:")
        print(binary_mask[0])
        new_image = image * binary_mask[0][..., np.newaxis]
        alpha = np.sum(new_image, axis=-1) > 0

        # Convert True/False to 0/255 and change type to "uint8" to match "na"
        alpha = np.uint8(alpha * 255)
        image_with_transparent_background = np.dstack((new_image,alpha ))
        # TODO: might need to return the base64 data back
        print("LOG - save segmentation data")
        return cv2.imwrite("../output/output.png", image_with_transparent_background)


model = Model()
app = FastAPI()


@app.post("/ping")
async def test_ping(req: Request) -> Response:
    base64 = req.base64
    return Response(text=base64)


@app.post("/sticker")
async def generate_sticker(req: Request) -> Response:
    base64 = req.base64
    link = req.link 
    return Response(text="true" if model.infer(link) == True else "false")

