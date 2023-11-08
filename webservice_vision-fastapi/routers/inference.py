from fastapi import APIRouter, File, UploadFile, Request
import numpy as np
import cv2

from schemas.sch_inference import InferenceModel



router = APIRouter(
  prefix='/Inference',
  tags=['VISION_API']
)


@router.post('/image')
async def get_file(request: Request, nmsthreshold: float = 0.3, file: UploadFile = File(...)):
    detector = request.app.state.model
    image_content = await file.read()
    imagen_np = np.frombuffer(image_content, np.uint8)
    imagen_opencv = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR) 

    # Realizar la inferencia en la imagen de manera asincrónica
    async def perform_inference():
        results = detector.detect_objects(imagen_opencv, 
                                          float(nmsthreshold), 
                                          debug=False)
        return InferenceModel.format_results(results)
    results = await perform_inference()

    return results






