from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from routers.home import router as home
from routers.inference import router as inference
from engine.object_detection import ObjectDetector


def run_app():

    load_dotenv("config/.env")
    detector = ObjectDetector(os.getenv("PATH_WEIGTH"), os.getenv("PATH_CFG"),
                              os.getenv("PATH_CLASSES"), os.getenv("FILTER_CLASS"))

    app = FastAPI()

    app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
    
    app.state.model = detector # https://stackoverflow.com/questions/71298179/fastapi-how-to-get-app-instance-inside-a-router
    app.include_router(home) 
    app.include_router(inference) 

    return app

if __name__ =="__main__":
    run_app()




# if __name__ == "__main__":
#     detector = ObjectDetector("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg", "dnn_model/classes.txt")
#     results = detector.detect_objects("images/prueba2.jpg", debug=True)
#     print(results)


# uvicorn main:run_app --host 0.0.0.0 --port 8008
# uvicorn main:run_app --reload


