# overview (~p187)
class PSPNet(nn.module):
	def __init__(self, n_classes):
		super(PSPNet, self).__init__()

		block_config = [3,4,6,3]
		img_size = 475
		img_size_8 = 60

		self.feature_conv = FeatureMap_convolution()

		self.feature_res_1 = ResidualBlockPSP(
			n_blocks=block_config[0], in_channels=128, mid_channels=64,
			out_channels=256, stride=1, dilation=1)

		self.feature_res_2 = ResidualBlockPSP(
			n_blocks=block_config[1], in_channels=256, mid_channels=128,
			out_channels=512, stride=2, dilation=1)

		self.feature_dilated_res_2 = ResidualBlockPSP(
			n_blocks=block_config[3], in_channels=1024, mid_channels=512,
			out_channels=2048, stride=1, dilation=4)
		
		self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[
			6, 3, 2, 1], height=img_size_8, n_classes=n_classes)
	
		self.decode_feature = DecodePSPFeature(
			height=img_size, width=img_size, n_classes=n_classes)
		
		self.aux = AuxilaryPSPlayers(
			in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)

	def forward(self, x):
		x = self.feature_conv(x)
		x = self.feature_res_1(x)
		x = self.feature_res_2(x)
		x = self.feature_dilated_res_1(x)

		output_aux = self.aux(x)

		x = self.feature_dilated_res_2(x)

		x= self.pyramid_pooling(x)
		output = self.decode_feature(x)

		return (output, output_aux)

# FeatureMap_convolution (~p191)

class conv2DBatchNormRelu(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
		 dilation, bias):
		super(conv2DBatchNormRelu, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels,
													kernel_size, stride, padding, dilation, bias=bias)
		self.batchnorm = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.batchnorm(x)
		outputs = self.relu(x)
		
		return outputs

### FeatureMap_convolution ###
class FeatureMap_convolution(nn.Module):
	def __init__(self):
		super(FeatureMap_convolution, self).__init()
		
		# conv1
		in_channels, out_channels, kernel_size, stride, spadding, dilation, bias = 
		3, 64, 3, 2, 1, 1, False
		self.cbnr_1 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size, stride, paddingm dilation, bias)

		#conv2
		in_channels, out_channels, kernel_size, stride, spadding, dilation, bias = 
		64, 64, 3, 1, 1, 1, False
		self.cbnr_2 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size, stride, paddingm dilation, bias)	

		#conv3
		in_channels, out_channels, kernel_size, stride, spadding, dilation, bias = 
		64, 128, 3, 1, 1, 1, False
		self.cbnr_3 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size, stride, paddingm dilation, bias)

		# max pool
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

	def forward(self, x):
		x = self.cbnr_1(x)
		x = self.cbnr_2(x)
		x = self.cbnr_3(x)
		ouputs = self.maxpool(x)
		return outputs

# ResidualBlockPSP (~p193)
class ResidualBlockPSP(nn.Sequential):
	def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride,
dilation):
		super(ResidualBlockPSP, self).__init__()

		# bottleNeckPSP
		self.add_module(
			"block1",
			bottleNeckPSP(in_channels, mid_channels,
										out_channels, stride, dilation)
		)
		# bottleNeckIdentifyPSP
		for i in range(n_blocks - 1):
			self.add_module(
				"block" + str(i+2),
				bottleNeckIdentifyPSP(
					out_channels, mid_channels, stride, dilation)
			)

# bottleNeckPSP, bottleNeckIdentifyPSP(p~197)
class conv2DBatchNorm(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
							dilation,bias):
		super(conv2DBatchNorm, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels,
													kernel_size, stride, padding, dilation, bias=bias)
		self.batchnorm = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = self.conv(x)
		outputs = self. batchnorm(x)

		return outputs

class bottleNeckPSP(nn.Module):
	def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
		super(bottleNeckPSP, self).__init__()

		self.cbr_1 = conv2DBatchNormRelu(
			in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
			dilation=1, bias=False)
		self.cbr_2 = conv2DBatchNormRelu(
			in_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation,
			dilation=dilation, bias=False)
		self.cb_3 = conv2DBatchNorm(
			mid_channels, out_channels, kernel_size=1, stride=1, padding=0,
			dilation=1, bias=False)

		self.cb_residual = conv2DbatchNorm(
			in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
			dukatuib=1, bias=False)

		self.relu =nn.ReLU(inplace=True)

	def forward(self, x):
		conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
		residual = self.cb_residual(x)
		return self.relu(conv + residual)

class bottleNeckIdentifyPSP(nn.Module):
	def __init__(self, in_channels, mid_channels, stride, dilation):
		super(bottleNeckIdentifyPSP, self).__init__()

		self.cbr_1 = conv2DBatchNormRelu(
			in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
			dilation=1, bias=False)
		self.cbr_2 = conv2DBatchNormRelu(
			in_channels, mid_channels, kernel_size=3, stride=1, padding=dilation,
			dilation=dilation, bias=False)
		self.cb_3 = conv2DBatchNorm(
			mid_channels, out_channels, kernel_size=1, stride=1, padding=0,
			dilation=1, bias=False)
		self.relu =nn.ReLU(inplace=True)

	def forward(self, x):
		conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
		residual = x
		return self.relu(conv + residual)

# Pyraomid Pooling (~p201)
class PyramidPooling(nn.Module):
	def __init__(self, in_channels, pool_sizes, height, width):
		super(PyramidPooling, self).__init__()

		# forward에서 필요한 resolution 정보
		self.height = height
		self.width = width
		
		# ouput channels for each conv
		out_channels = int(in_channels / len(pool_sizes))

		# for문이 좋지만 이해하기 쉽게 풀어서 구현
		pool_sizes: [6, 3, 2, 1]

		self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
		self.cbr_1 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0,
			dilation=1, bias=False)
		
		self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
		self.cb_2 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0,
			dilation=1, bias=False)

		self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
		self.cbr_3 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0,
			dilation=1, bias=False)

		self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
		self.cbr_4 = conv2DBatchNormRelu(
			in_channels, out_channels, kernel_size=1, stride=1, padding=0,
			dilation=1, bias=False)

	def forward(self, x):
		
		out1 = self.cbr_1(self.avpool_1(x))
		out1 = F.interpolate(out1, size=(
			self.height, self.width), mode="bilinear", align_corners=True)

		out2 = self.cbr_2(self.avpool_2(x))
		out2 = F.interpolate(out2, size=(
			self.height, self.width), mode="bilinear", align_corners=True)

		out3 = self.cbr_3(self.avpool_3(x))
		out3 = F.interpolate(out3, size=(
			self.height, self.width), mode="bilinear", align_corners=True)

		out4 = self.cbr_4(self.avpool_4(x))
		out4 = F.interpolate(out4, size=(
			self.height, self.width), mode="bilinear", align_corners=True)

		output = torch.cat([x, out1, out2, out3, out4], dim=1)

		return output

# Deocoder, AuxLoss (~p204)
class DecodePSPFeature(nn.Module):
	def __init__(self, height, width, n_classes):
		super(DecodePSPFeature, self).__init__()

	self.height= height
	self.width = width

	self.cbr = conv2DBatchNormRelu(
		in_channels=4096, out_channels=512, kernel_size=3, stride=1,
		padding=1, dilation=1, bias=False)
	self.dropout = nn.Dropout2d(p=0.1)
	self.classification = nn.Conv2d(
		in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		x = self.cbr(x)
		x = self.dropout(x)
		x = self.classifcation(x)
		output = F.interpolate(
			x, size=(self,height, self.width), mode="bilinear", align_corners=True)

		return output

class AuxiliaryPSPlayers(nn.Module):
	def __init__(self, in_channels, height, width, n_classes):
		super(AuxiliaryPSPlayers, self).__init()

		self.height = height
		self.width = width

		self.cbr = conv2DBatchNormRelu(
			in_channels=in_channels, out_channels=256, kernel_size=3, stride=1,
			padding=1, dilation=1, bias=False)
		self.drouput = nn.Dropout2d(p=0.1)
		self.classification = nn.Conv2d(
			in_channels=26, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		x = self.cbr(x)
		x = self.dropout(x)
		x = self.clasification(x)
		output = F.interpolate(
			x, size=(self.heightm self.width), mode="bilinear", align_corners=True)

		return output

net = PSPNet(n_classes=21)
net

batch_size = 2 
dummy_img = torch.rand(batch_size, 3, 475, 475)

outputs = net(dummy_img)
print(outputs)
		

		