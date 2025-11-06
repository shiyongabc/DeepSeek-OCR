
from transformers import AutoModel, AutoTokenizer
import torch
import os
from pathlib import Path
from paddleocr import PPStructureV3

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#model_name = 'deepseek-ai/DeepSeek-OCR'
model_name= '/dev/shm/deepseek-ocr'
def deepseek_ocr(image_file,output_path="output"):

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)

    # prompt = "<image>\nFree OCR. "
    #prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    #image_file = '/root/workspace/ocr/fp_page_1.png'
    #output_path = '/root/workspace/ocr/output/dir'
    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)


"""
deepseek-ocr解析发票图片的时候：发票号码和发票日期提取不出来
paddle-ocr能结构化提取全部内容
"""
def paddle_ocr(input):
    """
    input=输入文件的文件路径 可以是本地路径 也可以是http/https的地址
    """
    pipeline = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False
    )

    # For Image
    output = pipeline.predict(
        input,
    )

    # Visualize the results and save the JSON results
    for res in output:
        res.print()
        res.save_to_json(save_path="output")
