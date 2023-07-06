import onnxruntime
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# 加载模型
def load_model(model_path):
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(model_path, sess_options)
    return session

def preprocess_image(image):
    image = image.resize((128, 128), Image.ANTIALIAS)
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# 进行检测
def classify(filename, is_folder=False):
    model_path = "optimized.onnx"  # 替换为您的模型路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_path)
    session = load_model(model_path)

    if is_folder:
        image_files = os.listdir(filename)
        results = []
        for file_name in tqdm(image_files, desc="Classifying candidates"):
            image_path = os.path.join(filename, file_name)
            image = Image.open(image_path)
            width, _ = image.size
            num_regions = width // 100  # 计算第一行截取区域的个数

            max_true_conf = 0
            max_result = None

            for i in range(num_regions):
                left = i * 100
                right = left + 100
                image_region = image.crop((left, 0, right, 100))
                input_image = preprocess_image(image_region)

                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                output = session.run([output_name], {input_name: input_image})
                true_conf = float(output[0][0][1])
                bogus_conf = float(output[0][0][0])

                if true_conf > max_true_conf:
                    max_true_conf = true_conf
                    max_result = {
                        'file_name': file_name,
                        'true_conf': true_conf,
                        'bogus_conf': bogus_conf
                    }

            results.append(max_result)

        return results
    else:
        image_path = os.path.join(current_dir, filename)
        image = Image.open(image_path)
        width, _ = image.size
        num_regions = width // 100  # 计算第一行截取区域的个数

        max_true_conf = 0
        max_result = None

        for i in range(num_regions):
            left = i * 100
            right = left + 100
            image_region = image.crop((left, 0, right, 100))
            input_image = preprocess_image(image_region)

            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            output = session.run([output_name],{input_name: input_image})
            true_conf = float(output[0][0][1])
            bogus_conf = float(output[0][0][0])

            if true_conf > max_true_conf:
                max_true_conf = true_conf
                max_result = {
                    'true_conf': true_conf,
                    'bogus_conf': bogus_conf
                }

        return max_result