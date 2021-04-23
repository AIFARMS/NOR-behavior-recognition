import torch
import torch.nn.functional as F

class C3D_detector():

	def __init__(self, model, checkpoint):
		"""
		Load classifier from checkpoint
		"""
		model.load_state_dict(torch.load(checkpoint))
		self.model = model
		self.model.eval()

		self.action_names = ['explore', 'investigate']

	def detect(self, c3d_feature):
		"""
		Given C3D feature, detect action
		"""
		out = self.model(torch.from_numpy(c3d_feature))

		score, y_pred = F.softmax(out).max(dim=0)
		
		return self.action_names[y_pred], float(score)