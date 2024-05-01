from PIL import Image
import torch.utils.data as data

def read_image(path):
	try:
		img = Image.open(path).convert('RGB')
		return img

	except:
		print('Load Image Error!')
		return Image.new('RGB', (224,224), 'white')

class VireoLoader(data.Dataset):
	def __init__(self, img_path, transform):
		# self.path = path
		self.img_path = img_path
		self.transform = transform
		self.dataset = [img_path]

	def __getitem__(self, index):
		img = read_image(self.img_path)
		img = self.transform(img)
		return [img]

	def __len__(self):
		return len(self.dataset)
