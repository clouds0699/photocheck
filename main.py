import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from opencv_matching.template_matching import TemplateMatching

app = FastAPI()


class Photo(BaseModel):
    target_path: str
    template_path: str
    threshold: float = 0.8


@app.post("/match")
async def postdate(photo: Photo):
    # 读取目标图片
    target = cv2.imread(photo.target_path)
    # 读取模板图片
    template = cv2.imread(photo.template_path)

    if target is None or template is None:
        raise HTTPException(status_code=401, detail="img path is invalid")
    else:
        result = TemplateMatching(target, template, photo.threshold).find_all_results()
        if result is not None:
            return HTTPException(status_code=200,
                                detail="success")
        else:
            raise HTTPException(status_code=405,
                                detail="confidence is less than setting value,or target img is larger than template img")


if __name__ == "__main__":
    uvicorn.run(app='main:app', host='127.0.0.1', port=8100, reload=True)
