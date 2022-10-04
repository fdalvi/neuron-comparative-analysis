import cv2
#获取一张图片的宽高作为视频的宽高
image=cv2.imread('../figures/probeless_1.png')
# cv2.imshow("new window", image)   #显示图片
image_info=image.shape
height=image_info[0]
width=image_info[1]
size=(height,width)
print(size)
fps=30
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter('lca1.mp4', -1, 5, (width,height)) #创建视频流对象-格式一


for i in range(1,200):
    file_name = '../figures/lca_1_' + str(i) + '.png'
    image=cv2.imread(file_name)
    video.write(image)  # 向视频文件写入一帧--只有图像，没有声音
video.release()
cv2.waitKey()