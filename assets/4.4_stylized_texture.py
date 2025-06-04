'''
尝试用AI对不同语义分割区域进行风格迁移（未完成）
'''
import base64
from openai import OpenAI
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

client = OpenAI(
    api_key="sk-UyCHsUTcWaO0xZNYK7Rg4YIQ79dROSfdv2IOMer9s4nYYmLu",
    base_url="https://api.getgoapi.com/v1"
)

result = client.images.edit(
    model="gpt-image-1",
    background="opaque",
    quality="high",
    size="auto",
    image=[
        open("/data2/zyz/S3DIS/area_3/pano/rgb/camera_0ccf3c78ef354902b516c62ef8fb7cf1_lounge_2_frame_equirectangular_domain_rgb.png", "rb"),
        open("/data2/zyz/S3DIS/area_3/pano/semantic_pretty/camera_0ccf3c78ef354902b516c62ef8fb7cf1_lounge_2_frame_equirectangular_domain_semantic_pretty.png", "rb"),
        ],
    prompt="图片2是图片1的语义分割mask，要求随机交换不同mask区域的纹理风格，生成一张新的图片，并尽可能保持除纹理外的图像细节与图片1一致",

)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)


# 读取图片为PIL对象
image = Image.open(io.BytesIO(image_bytes))
# 转为numpy数组
image_np = np.array(image)

# 可视化
plt.imshow(image_np)
plt.axis('off')
plt.show()

# Save the image to a file
with open("composition.png", "wb") as f:
    f.write(image_bytes)