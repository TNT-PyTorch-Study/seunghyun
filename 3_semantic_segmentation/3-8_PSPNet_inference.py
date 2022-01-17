# inference (~p219)

from utils.dataloader import make_datapath_list, DataTransform

rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list,val_anno_list = make_datapath_list(
	rootpath=rootpath)

from utils.pspnet import PSPNet

net = PSPNet(n_classes=21)

state_dict = torch.load("./weights/pspnet50_30.pth",
												map_location={"cuda:0": "cpu"}
net.load_state_dict(state_dict)

print("네트워크 설정 완료")

# 추론 시작

image_file_path = "./data/cowboy-757575_640.jpg"
img = Image.open(image_file_path)
img_width, img)height = img.size
olt.imshow(img)
plt.show()

color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
transform = DataTransform(
	input_size=475, color_mean=color_mean, color_std=color_std)

# preprocess

anno_file_path = val_anno_list[0]
anno_class_img = Image.open(anno_file_path)
p_palette - anno_class_img.getpalette()
phase = "val"
img, anno_class_img.getpalette()
phase = "val"
img, anno_class_img = transform(phase, img, anno_class_img)

# 추론
net.eval()
x = img.unsqueeze(0)
outputs = net(x)
y = outputs[0]

y = y[0].detach().numpy()
y = np.argmax(y, axis=0)
anno_class_img = Image.fromarray(np.uint8(y), mode="p") #  475 X 475
anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST) # nearest neighbor
anno_class_img.putpalette(p_palette)
plt.imshow(anno_class_img)
plt.show()

trans_img = Image.new('RGBA', anno_class_img.size, (0,0,0,0))
anno_class_img = anno_class_img.convert('RGBA')

for x in range(img_width):
	for y in range(img_height):
		pixel = anno_class_img.getpixel((x,y))
		r, g, b, a = pixel

		if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
			continue
		else:
			trans_img.putpixel((x,y), (r,g,b,150))

img = Image.open(image_file_path)
result = Image.alpha_composite(img.conver('RGBA'), trans_img)
plt.imshow(result)
plt.show()
 
