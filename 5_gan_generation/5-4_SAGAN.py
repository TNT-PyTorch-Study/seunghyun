# intro
X = X.view(X.shape[0],X.shape[1], X.shape[2]*X.shape[3])

X_T = x.permute(0, 2,1)
S = toch.bmm(X_T, X)

m = nn.Softmax(dim=-2)
attention_map_T = m(S)
attention_map = attention_map_T.permute(0, 2, 1)

o = torch.bmm(X, attention_map.permute(0,2,1))

query_conv = nn.Conv2d(
	in_channels=X.shape[1], out_channels=X.shape[1]//8, kernel_size=1)
key_conv = nn.Conv2d(
	in_channels=X.shape[1], out_channels=X.shape[1]//8, kernel_size=1)
value_conv = nn.Conv2d(
	in_channels=X.shape[1], out_channels=X.shape[1], kernel_size=1)

proj_query = query_conv(X.view(
	X.shape[0], -1, X.shape[2]*X.shape[3])
proj_query = proj_query.permute(0,2,1)
proj_key = key_conv(X).view(
	X.shape[0], -1, X.shape[2]*X.shape[3])

S = torch.bmm(proj_query, proj_key)

m = nn.Softmax(dim=-2)
attention_map_T = m(S)
attention_map = attention_map_T.permute(0,2,1)

proj_value = value_conv(X).view(
	X.shape[0], -1, X.shape[2]*X.shape[3])
o = torch.bmm(proj_value, attention_map.permute(
	0, 2, 1))


# self-attention
class Self_Attention(nn.Module):
	def __init__(self, in_dim):
		super(Self_Attention, self).__init()

		self.query_conv = nn.Conv2d(
			in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
		self.key_conv = nn.Conv2d(
			in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
		self.value_conv = nn.Conv2(
			in_channels=in_dim, out_channels=in_dim, kernel_size=1)

		self.softmax = nn.Softmax(dim=-2)

		self.gamma = nn.Parameter(torch.zers(1))

	def forward(self, x):

		X = x

		proj_query = self.query_conv(X).view(
			X.shape[0], -1, X.shape[2]*X.shape[3])
		proj_query = proj_query.permute(0, 2, 1)
		proj_key = self.key_conv(X).view(
			X.shape[0], -1, X.shape[2]*X.shape[3])

		S = torch.bmm(proj_query, proj_key)

		attention_map_T = self.softmax(S)
		attention_map = attention_map.permute(0, 2, 1)

		proj_value = self.value_conv(X).view(
			X.shape[0], -1, X.shape[2]*X.shape[3])
		o = torch.bmm(proj_value, attention_map.permute(
			0, 2, 1))
		
		o = o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
		out = x+self.gamma*o

		return out, attention_map

# generator	

class Generator(nn.Module):

	def __init__(self, z_dim=20, image_size=64):
		super(Generator, self).__init__()

		self.layer1 = nn.Sequential(
			nn.utils.spectral_norm(nn.ConvTranspose2d(z_dim, image_size * 8,
				kernel_size=4, stride)),
			nn.BatchNorm2d(image_Size * 8),
			nn.ReLU(inplace=True))

		self.layer2 = nn.Sequential(
			nn.utils.spectral_norm(nn.ConvTranspose2d(image_size * 8, image_size * 4,
				kernel_size=4, stride=2, padding=1)),
		
		nn.BatchNorm2d(image_size * 4),
		nn.ReLU(inplace=True))
		
		self.layer3 = nn.Sequential(
			nn.utils.spectral_norm(nn.ConvTranspose2d(image * 4, image_size * 2,
				kernel_size=4, stride2, padding=1)),
			nn.BatchNorm2d(image_size * 2), nn.ReLU(inplace=True))

		self.self_attention1 = Self_Attention(in_dim=image_size * 2)

		self.layer4 = nn.Sequential(
			nn.utils.spectral_norm(nn.ConvTranspose2d(image_size * 2, image_size,
				kernel_size=4, stride=2, padding=1)),
			nn.BatchNorm2d(image_size),
			nn.ReLU(inplace=True))

		self.sefl_attention2 = Self_Attention(in_dim=image_size)

		self.last = nn.Sequential(
			nn.ConvTranpose2d(image_size, 1, kernel_size=4,
				stide=2, padding=1),
			nn.Tanh())

	def forward(self, z):
		out = self.layer1(z)
		out = self.layer2(out)
		out = self.layer3(out)
		out, attention_map1 = self.sef_attntion1(out)
		out = self.layer4(out)
		out, attention_map2 = self.self_attention2(out)
		out = self.last(out)

		return out, attention_map1, attention_map2

# discriminator

class Discriminator(nn.Module):
	def __init__(self, z_dim=20, image_size=64):
		super(Discriminator, self).__init__()

		self.layer1 = nn.Sequential(
			nn.utils.spectral_norm(nn.Conv2d(1, image_size, kernel_size=4,
				stride=2, padding=1)),
			nn.LeakyReLU(0.1, inplace=True))

		self.layer2 = nn.Sequential(
			nn.utils.spectral_norm(nn.Conv2d(image_size, image_size*2, kernel_size=4,
				stirde=2, padding=1)),
			nn.LeakyReLU(0.1, inplace=True))

		self.layer3 = nn.Sequential(
			nn.utils.spectral_norm(nn.Conv2d(image_Size*2, image_size*4, kernel_size=4,
				stride=2, padding=1)),
			nn.LeakyReLU(0.1, inplace=True))

		self.self_attntion1 = Self_Attention(in_dim=image_size*4)

		self.layer4 = nn.Sequential(
			nn.utils.spectral_norm(nn.Conv2d(image_size*4, image_size*8, kernel_size=4,
				stride=2, padding=1)),
			nn.LeakyReLU(0.1, inplace=True))

		self.self_attntion2 = Self_Attention(in_dim=image_size*8)

		self.last =nn.Conv2d(image_size*8, 1, kernel_size=4, stride=1)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out, attention_map1 = self.self_attntion1(out)
		out = self.layer4(out)
		out, attention_map2 = self.self_attntion2(out)
		out = self.last(out)
		
		return out, attention_map1, attention_map2


# dataloader
def train_model(G, D, dataloader, num_epochs):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("using device: ", device)

	g_lr, d_lr = 0.0001, 0.0004
	beta1, beta2 = 0.0, 0.9
	
	g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
	d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

	criterion = nn.BCEWithLogitsLoss(reduciton='mean')

	z_dim = 20
	mini_batch_size =64

	G.to(device)
	D.to(device)

	G.train()
	D.train()

	torch.backends.cudnn.benchmark = True

	num_train_imgs = len(dataloader.dataset)
	batch_size = dataloader.batch_size

	iteration = 1 
	logs = []

	for epoch in range(num_epochs):
		t_epoch_start = time.time()
		epoch_g_loss =0.0
		epoch_d_loss =0.0
		
		print('----------')
		print('Epoch {}/{}'.format(epoch, num_epochs))
		print('----------')
		print(' (train) ')

		for imges in dataloder:
			if imges.size()[0] == 1:
				continue
			imges = imges.to(device)

			mini_batch_size = imges.size()[0]
			
			d_out_real, _, _ = D(imges)
			input_z = torch.randn(mini_batch_size, z_dim).to(device)
			input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
			fake_images, _, _ = G(input_z)
			d_out_fake, _, _ = D(fake_images)

			d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

			d_loss = d_loss_real + d_loss_fake

			g_optimizer.zero_grad()
			g_optimizer.zero_grad()

			d_loss.backward()
			d_optimizer.step()

			input_z = torch.randn(mini_batch_size, z_dim).to(device)
			input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
			fake_images, _, _ = G(input_z)
			d_out_fake, _, _ = D(fake_images)

			g_loss = - d_out_fake.mean()

			g_optimizer.zero_grad()
			d_optimzier.zero_grad()
			g_loss.backward()
			g_optimzier.step()

			epoch_d_loss += d_loss.item()
			epoch_g_loss += g_loss.item()
			iteration += 1 
	t_epoch_fimish =time.time()
	print('----------')
	print('epocj {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f}'.format(
		epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
	print('timer: {:.4f} sec.' .format(t_epoch_finish - t_epoch_start))
	t_epoch_start = time.time()

	return G,D

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant-(m.bias.data, 0)

G.apply(weights_init)
D.apply(weights_init)

num_epochs = 300
G_update, D_update = train_model(
	G, D, dataloader =train_dataloader, num_epochs=num_epochs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 8
z_dim = 20
fixed_z = torch.randn(batch_size, z_dim)
fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

G_update.eval()
fake_images, am1, am2 = G_update(fixed_z.to(device))

batch_iterator = iter(train_dataloader)
imges = next(batch_iterator)

fig = plt.figure(figsize=(15, 6))

for i in range(0, 5):
	plt.subplot(2, 5, i+1)
	plt.imshow(imges[i][0].cpu().detach().numpy(), 'gray')

	plt.subplot(2, 5, 5+i+1)
	plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')


fig = plt.figure(figsize=(15, 6))
for i in range(0, 5):
	plt.subplot(2, 5, i+1)
	plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')

	plt.subplot(2, 5, 5+i+1)
	am = am1[i].view(16, 16, 16, 16)
	am = am[7][7]
		plt.imshow(am.cpu().detach().numpy(), 'Reds')