# dataloader and network
from utils.dataloader import make_datapath_list, DataTransform, COCOkeypointsDataset

train_img_list, train_mask_list, val_mask_list, train_meta_list,
val_meta_list = make_datapath_list(
	rootpath="./data/")

#dataset
train_dataset = COCOkeypointsDataset(
	val_img_list, val_mask_list, val_meta_list, phase="train", transform=DataTransform())

# validation skipped

batch_size = 32

train_dataloader = data.DataLoader(
	train_dataset, batch_size=batch_size, shuffle=True)

dataloaders_dict = {"train": train_dataloader, "val": None}

# create openpose_net isntanced
from utils.openpose_net improt OpenPoseNet
net = OpenPoseNet()

# loss function
class OpenPoseLoss(nn.Module):
	def __init__(self):
		super(OpenPoseLoss, self).__init__()

	def forward(self, saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask):
		total_loss = 0

		# each stage
		for j in range(6):
			pred1 = saved_for_loss[2 * j] * paf_mask
			gt1 = paf_target.float() * paf_mask

			pred2 = saved_for_loss[2 * j + 1] * heat_mask
			gh2 = heatmap_target.float()*heat_mask
	
			total_loss += F.mse_loss(pred1, gt1, reduction='mean') + \
				F.mse_loss(pred2, gt2, reduction='mean')

		return total_loss

criterion = OpenPoseLoss()

# train
optimizer = optim.SGD(net.parameters(), lr=1e-2,
											momentum=0.9
											weight_decay=0.0001)

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("device: ", device)

	net.to(device)

	torch.backends.cudnn.benchmark = True

	num_train_imgs = len(dataloaders_dict["train"].dataset)
	batch_size = dataloaders_dict["train"].batch_size

	iteration = 1

	for epoch in range(num_epochs):
		t_epoch_start = time.time()
		t_iter_start = time.time()
		epoch_train_loss = 0.0
		epoch_val_loss = 0.0

		print('--------------')
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('--------------')

		for phase in ['train', 'val']:
			if phase == 'train':
				net.train()
				optimizer.zero_grad()
				print(' (train) ')
			
			for imges, heatmap_target, heat_mask, paf_target, paf_mask in
					dataloaders_dict[phase]:
				if imges.size()[0] == 1:
					continue
				
				imges = imges.to(device)
				heatmap_target = hetamap_target.to(device)
				heat_mask = heat_mask.to(device)
				paf_target = paf-target.to(device)
				paf_mask = paf_mask.to(device)
	
				optimizer.zero_grad()
	
				with torch.set_grad_enabled(phase == 'train'):
					_, saved_for_loss = net(imges)
			
					loss = criterion(saved_for_loss, heatmap_target,
														heat_mask, paf_target, paf_mask)
					del saved_for_loss

					if phase == "train":
						loss.backward()
						optimizer.step()

						if (iteration % 10 == 0):
							t_iter_finish = time.time()
							duration = t_iter_finish - t_iter_start
							print('repeat {} || Loss: {:.4f} || 10iter: {:.4f}
								sec.'.format(
								iteration, loss.item()/batch_size, duration))
							print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
							t_epoch_start = time.time()
						torch.save(net.state_dict(), 'weights/openpose_net_' + 
												str(epoch+1) + '.pth')