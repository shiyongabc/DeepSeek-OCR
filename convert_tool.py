import fitz  # PyMuPDF
import os
def pdf_to_images_fitz(pdf_path, output_dir=None, zoom=2):
    """
    使用fitz库将PDF转换为图片
    :param pdf_path: PDF文件路径
    :param output_dir: 输出目录
    :param zoom: 缩放因子，控制图片质量
    :return: 生成的图片文件路径列表
    """
    if output_dir is None:
        output_dir = os.path.dirname(pdf_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_document = fitz.open(pdf_path)
    saved_files = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # 设置转换矩阵（控制分辨率）
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"{output_dir}/{base_name}_page_{page_num + 1}.png"

        # 保存图片
        pix.save(output_path)
        saved_files.append(output_path)
        print(f'已保存: {output_path}')

    pdf_document.close()
    return saved_files
