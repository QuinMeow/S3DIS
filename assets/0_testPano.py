'''
测试全景图生成
'''
import cv2
import py360convert

# 读取360°全景图像（equirectangular 格式）
img = cv2.imread('/data2/zyz/S3DIS/area_3/pano/rgb/camera_0ccf3c78ef354902b516c62ef8fb7cf1_lounge_2_frame_equirectangular_domain_rgb.png')

# 设置视场角和观察角度
fov = 65       # 视场角，单位度
theta = 0     # 水平角（yaw），单位度
phi = -3        # 垂直角（pitch），单位度

# 使用py360convert将全景图转换为透视图
persp_img = py360convert.e2p(img, fov, theta, phi, [512,512])
print(persp_img.dtype)
# 显示结果
cv2.imshow('Perspective View', persp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
