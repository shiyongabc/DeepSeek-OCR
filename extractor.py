import pdfplumber
import re
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json
import os
class Extractor:
    def __init__(self):
        self.encoding="utf8"
        self.fp_patterns = {
            # block_label+'_'+'patterns' 作为key 匹配对应的正则表达式
            'text_patterns':{
                'invoice_number': r'发票号码[:：]\s*(\d+)',
                'invoice_date': r'开票日期[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2})',
            },
            'table_patterns': {
                'purchaser_name': r'名称[:：]\s*([^\s]+)',
                'amount': r'\(小写\)￥\s*([\d,]+\.?\d*)',
                'purchaser_tax_id': r'纳税人识别号[:：]\s*([\d\w]+)',
                'seller_name': r'销售方信息</td><td rowspan="2">名称[:：]\s*([^\s]+)',
                'seller_tax_id': rf'{re.escape("销售方信息")}[^。]*?纳税人识别号[：:\s]*([a-zA-Z0-9]{{10,20}})',
            },
            'vision_footnote_patterns': {
                'drawer': r'开票人[:：]\s*([^\s]+)'
            },


        }
        self.idcard_patterns = {
            'name': r'姓名[:：]\s*(\d+)',
            'sex': r'性别[:：]\s*(\d+)',
            'birth': r'出生[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2})',
            'address': r'地址[:：]\s*([^\s]+)',
            'id_card': r'公民身份证号码\s*([\d,]+\.?\d*)',
        }
        #营业执照正则表达式
        self.register_patterns = {
            'invoice_number': r'发票号码[:：]\s*(\d+)',
            'invoice_date': r'开票日期[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2})',
            'purchaser_name': r'名称[:：]\s*([^\s]+)',
            'amount': r'小写(.*)\s*([\d,]+\.?\d*)',
            #'tax_amount': r'税额\s*([\d,]+\.?\d*)',
            #'total_amount': r'合计[:：]\s*([\d,]+\.?\d*)',
            'seller_tax_id': r'纳税人识别号[:：]\s*([\d\w]+)',

        }
    def extract_from_json(self, pdf_path: str,ocr_type:int) -> Dict:
        """
        从PDF文件中提取发票信息
        file_type 文件类型 1 pdf 2 txt/md
        ocr_type 1 发票识别 2 身份证 3 营养执照
        """
        invoice_data = {}
        try:
            #读取json   paddleocr-structure模型识别的结果
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"文件不存在: {pdf_path}")
            try:
                with open(pdf_path, 'r', encoding=self.encoding) as f:
                    data = json.load(f)
                    print("data=",data)
                    invoice_data=self.parserFromOcrJson(data,ocr_type)
            except UnicodeDecodeError:
                raise UnicodeDecodeError(f"文件编码错误，请检查文件是否为 {self.encoding} 编码")
        except Exception as e:
            print(f"处理PDF文件时出错: {e}")

        return invoice_data
    def parserFromOcrJson(self,data:Dict,ocr_type:int)->Dict:
        result_data={}
        parsing_res_list=data["parsing_res_list"]
        for item in parsing_res_list:
            print("item=",item)
            block_label=item["block_label"]
            block_content=item["block_content"]
            if block_label=="text" or block_label=="table" or block_label=="vision_footnote":
                invoice_data = self._parse_text(block_content, block_label + "_" + "patterns",ocr_type)
                #合并上一个返回的字典
                result_data.update(invoice_data)

        return result_data
    def _parse_text(self, text: str,pattern_key:str,ocr_type:int) -> Dict:
        """
        解析发票文本内容
        """
        invoice_data = {}
        patterns={}
        if ocr_type==1:
            patterns=self.fp_patterns[pattern_key]
        elif ocr_type==2:
            patterns = self.idcard_patterns[pattern_key]
        elif ocr_type==3:
            patterns = self.register_patterns[pattern_key]
        for field, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                print("match.group(0)=",match.group(0))
                print("match.group(1)=", match.group(1))
                invoice_data[field] = match.group(1)
            else:
                invoice_data[field] = None
        return invoice_data


def extract_tax_info_from_complex_text(text, target_chinese_string):
    """
    从复杂文本中提取特定中文字符串后的纳税人识别号

    参数:
        text: 完整文本
        target_chinese_string: 目标中文字符串

    返回:
        提取到的纳税人识别号
    """
    # 构建更复杂的正则表达式
    # 匹配：目标中文字符串 + 任意中文字符 + 纳税人识别号 + 营业执照号
    pattern = rf'{re.escape(target_chinese_string)}[^。]*?纳税人识别号[：:\s]*([a-zA-Z0-9]{{10,20}})'

    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    return None


def test_complex_register():
    """主函数，演示使用方法"""
    # 示例文本

    # 复杂文本提取示例
    complex_text = """
    在某某集团公司的年度报告中，首先介绍了公司的基本情况，
    包括公司名称：某某集团有限公司，注册地址：北京市海淀区，
    随后详细说明了税务相关信息，纳税人识别号：91110105MA01XYZ123，
    这是企业的重要标识信息。
    """

    print("复杂文本提取示例:")
    target_string = "某某集团有限公司"
    result = extract_tax_info_from_complex_text(complex_text, target_string)
    if result:
        print(f"✅ 从'{target_string}'后提取到的纳税人识别号: {result}")
    else:
        print("❌ 未找到匹配的信息")


def main():
    extractor = Extractor()

    # 单文件提取示例
    #pdf_path = "fp.pdf"  # 替换为实际文件路径
    pdf_path = "output/fp_res.json"  # 替换为实际文件路径
    results = extractor.extract_from_json(pdf_path,1)

    print("results=",results)

if __name__ == "__main__":
    main()
    #test_complex_register()
