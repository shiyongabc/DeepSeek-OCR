import os
import cv2
import re
import json
import numpy as np
from datetime import datetime
import paddle
from PIL import Image, ImageDraw, ImageFont


class InvoiceExtractor:
    def __init__(self, use_gpu=True):
        """
        初始化发票提取器
        Args:
            use_gpu: 是否使用GPU加速
        """
        # 配置GPU环境
        if use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            if paddle.is_compiled_with_cuda():
                paddle.device.set_device('gpu')
                print("GPU加速已启用")
            else:
                print("GPU不可用，使用CPU模式")
                use_gpu = False

        # 初始化PaddleOCR（兼容不同版本）
        try:
            from paddleocr import PaddleOCR
            self.ocr_engine = PaddleOCR(
                use_textline_orientation=True,
                lang='ch',
                use_gpu=use_gpu,
                gpu_mem=2000,
                text_detection_model_dir='ch_PP-OCRv4_det',
                text_recognition_model_dir='ch_PP-OCRv4_rec'
            )
            print("PaddleOCR初始化成功")
        except ImportError as e:
            print(f"PaddleOCR导入失败: {e}")
            print("请尝试: pip install paddleocr>=2.6.0")
            return

        # 发票字段正则表达式模式
        self.patterns = {
            'invoice_code': [
                r'发票代码[:：]\s*([0-9]{10,12})',
                r'代码[:：]\s*([0-9]{10,12})'
            ],
            'invoice_number': [
                r'发票号码[:：]\s*([0-9]{8,10})',
                r'号码[:：]\s*([0-9]{8,10})'
            ],
            'issue_date': [
                r'开票日期[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日)',
                r'日期[:：]\s*(\d{4}-\d{1,2}-\d{1,2})',
                r'(\d{4}年\d{1,2}月\d{1,2}日)'
            ],
            'amount': [
                r'价税合计[（(]小写[）)]\s*[:：]?\s*(¥|￥|人民币)?\s*([0-9,]+\.?\d{0,2})',
                r'小写\s*[:：]?\s*(¥|￥|人民币)?\s*([0-9,]+\.?\d{0,2})'
            ],
            'seller_name': [
                r'销售方[:：]\s*(.*?)(?:\n|$)',
                r'销方名称[:：]\s*(.*?)(?:\n|$)'
            ],
            'buyer_name': [
                r'购买方[:：]\s*(.*?)(?:\n|$)',
                r'购方名称[:：]\s*(.*?)(?:\n|$)'
            ],
            'tax_amount': [
                r'税额[:：]\s*(¥|￥|人民币)?\s*([0-9,]+\.?\d{0,2})'
            ],
            'check_code': [
                r'校验码[:：]\s*([0-9A-Z]{16,20})'
            ],
            'seller_tax_number': [
                r'纳税人识别号[:：]\s*([0-9A-Z]{15,20})'
            ]
        }

    def extract_text_from_image(self, image_path):
        """
        从图片中提取文本
        Args:
            image_path: 图片路径
        Returns:
            list: 提取的文本结果
        """
        print(f"开始提取图片文本: {image_path}")

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 执行OCR识别
        start_time = datetime.now()
        result = self.ocr_engine.ocr(img, cls=True)
        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"文本提取完成，耗时: {processing_time:.2f}秒")

        # 解析结果
        extracted_data = []
        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    extracted_data.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence
                    })

        return extracted_data, processing_time

    def parse_invoice_fields(self, extracted_data, img_shape):
        """
        解析发票字段
        Args:
            extracted_data: 提取的文本数据
            img_shape: 图像形状
        Returns:
            dict: 解析后的发票数据
        """
        invoice_data = {}

        for item in extracted_data:
            text = item['text']
            bbox = item['bbox']
            confidence = item['confidence']

            # 计算文本框中心位置
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x_center = np.mean(x_coords)
            y_center = np.mean(y_coords)

            # 匹配发票字段
            for field, patterns in self.patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match:
                        value = match.group(1)
                        # 如果同一字段有多个匹配，保留置信度更高的
                        if field not in invoice_data or confidence > invoice_data[field]['confidence']:
                            invoice_data[field] = {
                                'value': value,
                                'confidence': confidence,
                                'position': (x_center, y_center)
                            }

        return invoice_data

    def save_results(self, results, output_format='json'):
        """
        保存提取结果
        Args:
            results: 提取结果
            output_format: 输出格式
        Returns:
            str: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_format == 'json':
            filename = f"invoice_results_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"结果已保存: {filename}")
        return filename


def main():
    """
    主函数 - 发票信息提取演示
    """
    print("=" * 60)
    print("        PaddleOCR 发票信息提取系统")
    print("=" * 60)

    try:
        # 初始化提取器（启用GPU）
        extractor = InvoiceExtractor(use_gpu=True)

        # 示例图片路径
        image_path = "fp.png"

        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"示例图片不存在: {image_path}")
            print("请将您的发票图片命名为 'invoice_sample.jpg' 并放在当前目录")
            print("或者修改代码中的 image_path 变量")
            return

        # 提取文本信息
        extracted_data, processing_time = extractor.extract_text_from_image(image_path)

        # 解析发票字段
        img = cv2.imread(image_path)
        invoice_data = extractor.parse_invoice_fields(extracted_data, img.shape)

        # 可视化结果
        # vis_path = extractor.visualize_results(image_path, extracted_data)

        # 打印结果
        print("\n" + "=" * 40)
        print("           提取结果")
        print("=" * 40)

        for field, data in invoice_data.items():
            display_name = extractor.get_field_display_name(field)
            print(f"{display_name}: {data['value']} (置信度: {data['confidence']:.2f})")

        print(f"\n处理统计:")
        print(f"- 处理时间: {processing_time:.2f}秒")
        print(f"- 识别字段数: {len(invoice_data)}")

        # 保存结果
        results = {
            'extracted_data': invoice_data,
            'processing_time': processing_time,
            'total_fields': len(invoice_data)
        }

        extractor.save_results(results, 'json')

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        print("\n解决方案:")
        print("1. 确保已安装正确版本的PaddleOCR: pip install paddleocr>=2.6.0")
        print("2. 检查CUDA和cuDNN是否安装正确")
        print("3. 验证PaddlePaddle-GPU版本")

    def get_field_display_name(self, field_key):
        """
        获取字段显示名称
        Args:
            field_key: 字段键名
        Returns:
            str: 中文显示名称
        """
        field_names = {
            'invoice_code': '发票代码',
            'invoice_number': '发票号码',
            'issue_date': '开票日期',
            'amount': '金额',
            'seller_name': '销售方名称',
            'buyer_name': '购买方名称',
            'tax_amount': '税额',
            'check_code': '校验码',
            'seller_tax_number': '销售方税号'
        }
        return field_names.get(field_key, field_key)


if __name__ == "__main__":
    main()
