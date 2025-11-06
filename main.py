# coding: utf-8
import uvicorn
from fastapi import FastAPI, Request,UploadFile,File
from pathlib import Path
import logging
import json
import convert_tool
import ocr
from extractor import Extractor

logger = logging.getLogger(__name__)
app = FastAPI()


@app.post("/ai/ocr")
async def ocr(request:Request):# 限制10MB
    request_body = await request.body()
    decoded_str = request_body.decode('utf-8')
    print("decoded_str=%s" % decoded_str)
    logger.info(f"decoded_str==: {decoded_str}")
    requestJson = json.loads(decoded_str)
    # ocr_type 1 发票识别 2 身份证 3 营养执照
    # {"ocr_type":"1","url":"图片地址"}
    # logging.info("requestJson: {}".format(requestJson))
    print("requestJson=%s" % requestJson)
    ocr_type=requestJson["ocr_type"]
    url = requestJson["url"]

    path_obj = Path(url)
    # 获取后缀名（包含点号）
    extension_with_dot = path_obj.suffix
    #获取不带后缀的文件名
    filename_pathlib = path_obj.name
    print("extension_with_dot=",extension_with_dot)
    print("filename_pathlib=", filename_pathlib)
    extension_without_dot=filename_pathlib.replace(extension_with_dot,"")
    #ocr识别 保存json文件
    ocr.paddle_ocr(url)
    #解析保存的结果文件json
    extractor = Extractor()
    extractor_path = "output/"+extension_without_dot+"_res.json"  # 替换为实际文件路径
    results = extractor.extract_from_json(extractor_path,ocr_type)

    return results

if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


