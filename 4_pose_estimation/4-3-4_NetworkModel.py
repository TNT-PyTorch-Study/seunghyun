class OpenPoseNet(nn.Module):
	def __init__(self):
		super(OpenPoseNet, self).__init__()

		# feature module
		self.model0 = Openpose_Feature()

		# Stage Module
		# PAFs
		self.model1_1 = make_OpenPose_block('block1_1')
		self.model2_1 = make_OpenPose_block('block2_1')
		self.model3_1 = make_OpenPose_block('block3_1')
		self.model4_1 = make_OpenPose_block('block4_1')
		self.model5_1 = make_OpenPose_block('block5_1')
		self.model6_1 = make_OpenPose_block('block6_1')

		# confidence heatmap 측
		self.model1_2 = make_OpenPose_block('block1_2')
		self.model2_2 = make_OpenPose_block('block2_2')
		self.model3_2 = make_OpenPose_block('block3_2')
		self.model4_2 = make_OpenPose_block('block4_2')
		self.model5_2 = make_OpenPose_block('block5_2')
		self.model6_2 = make_OpenPose_block('block6_2')

	def forward(self, x):
		
		# feature
		out1 = self.model0(x)
		
		# stage1
		out1_1 = self.model1_1(out1) # PAFs
		out1_2 = self.model1_2(out1) # confidence heatmap
		# stage2
		out2 = torch.cat([out1_1, out1_2, out1], 1)
		out2_1 = self.model2_1(out2)
		out2_2 = self.model2_2(out2)
		# stage3
		out3 = torch.cat([out2_1, out2_2, out2], 1)
		out3_1 = self.model3_1(out2)
		out3_2 = self.model3_2(out2)
		# stage4
		out4 = torch.cat([out3_1, out3_2, out3], 1)
		out4_1 = self.model4_1(out2)
		out4_2 = self.model4_2(out2)
		# stage5
		out5 = torch.cat([out4_1, out4_2, out4], 1)
		out5_1 = self.model5_1(out2)
		out5_2 = self.model5_2(out2)
		# stage6		
		out6 = torch.cat([out5_1, out5_2, out5], 1)
		out6_1 = self.model6_1(out2)
		out6_2 = self.model6_2(out2)

		# for loss computation
		saved_for_loss = []
		saved_for_loss.append(out1_1)
		saved_for_loss.append(out1_2)
		saved_for_loss.append(out2_1)
		saved_for_loss.append(out2_2)
		saved_for_loss.append(out3_1)
		saved_for_loss.append(out3_2)
		saved_for_loss.append(out4_1)
		saved_for_loss.append(out4_2)
		saved_for_loss.append(out5_1)
		saved_for_loss.append(out5_2)
		saved_for_loss.append(out6_1)
		saved_for_loss.append(out6_2)

		return (out6_1, out6_2), saved_for_loss

        # feature & stage module
        class OpenPose_Feature(nn.Module):
	def __init__(self):
		super(OpenPose_Feature, self).__init__()

		vgg19 = torchvision,models.vgg19(pretrained=True)
		model = {}
		model['block0'] = vgg19.features[0:23] # VGG-19 최초 10개 합성곱

		model['block0'].add_module("23", torch.nn.Conv2d(
			512, 256, kernel_size=3, stride=1, padding=1))
		model['block0'].add_module("24", torch.nn.ReLU(inplace=True))
		model['block0'].add_module("25", torch.nn.Conv2d(
			256, 128, kernel_size=3, stride=1, padding=1))
		model['block0'].add_module("26", torch.nn.ReLU(inpace=True))

		self.model = model['block0']

	def forward(self, x):
		outputs = self.model(x)
		return outputs

	# stage modules
	def make_OpenPose_block(block_name):
		
		blocks = {}
		blocks['block1_1'] = [{'conv5_1_CPM_L1': [128,128,3,1,1]},
													{'conv5_2_CPM_L1': [128,128,3,1,1]},
													{'conv5_3_CPM_L1': [128,128,3,1,1]},
													{'conv5_4_CPM_L1': [128,512,1,1,0]},
													{'conv5_5_CPM_L1': [512,38,1,1,0]}]

		blocks['block1_2'] = [{'conv5_1_CPM_L2': [128,128,3,1,1]},
													{'conv5_2_CPM_L2': [128,128,3,1,1]},
													{'conv5_3_CPM_L2': [128,128,3,1,1]},
													{'conv5_4_CPM_L2': [128,512,1,1,0]},
													{'conv5_5_CPM_L2': [512,19,1,1,0]}]	

		# stage 2~6
		for i in range(2, 7):
			blocks['block%d_1' % 1] = [
				{'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
				{'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
				{'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
				{'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
				{'Mconv5_stage%d_L1' % i: [185, 128, 7, 1, 3]},
				{'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
				{'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]},
			]
			blocks['block%d_2' % 1] = [
				{'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
				{'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
				{'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
				{'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
				{'Mconv5_stage%d_L2' % i: [185, 128, 7, 1, 3]},
				{'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
				{'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]},
			]

		cfg_dict = blocks[block_name]

		layers = []

		for i in range(len(cfg_dict)):
			for k, v in cfg_dict[i].items():
				if 'pool' in k:
					layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
				else:
					conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
															kernel_size=v[2], stride=v[3],
															padding=v[4])
					layers += [conv2d, nn.ReLU(inplace=True)]
		
		# 마지막에는 ReLU필요없음
		net = nn.Sequential(*layers[:-1])
	
    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

        net.apply(_initialize_weights_norm)

        return net

# check

net = OpenPoseNet()
net.train()

batch_size = 2
dummy_img = torch.rand(batch_size, 3, 368, 368)

outputs = net(dummy_img)
print(outputs)

# Tensorboard
from utils.openpose_net import OpenPoseNet

net = OpenPoseNet()
net.train()

# tensorboardX
from tensonboardX import SummaryWriter

writer = SummaryWriter("./tbX/")

batch_size = 2
dummy_img = torch.rand(batch_size, 3, 368, 368)

writer.add_graph(net, (dummy_img,))
writer.close()

# prompt
tensorboard --logdir="./tbX/"
