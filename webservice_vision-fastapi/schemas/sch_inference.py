from pydantic import BaseModel
from typing import Optional
from fastapi import UploadFile, File
import json

class InferenceModel(BaseModel):
    file: UploadFile = File(...)
    publised: Optional[float] = 0.3

    @staticmethod
    def format_results(results):
        formatted_data = {int(key): [{'class_name': item['class_name'], 'score': float(item['score'])} for \
                                    item in value] for key, value in results.items()}
        json_data = json.dumps(formatted_data, separators=(',', ':'), ensure_ascii=False)
        json_data = json_data.replace('"', "'")
        return json_data