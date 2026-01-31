import glob
import os
import pickle
import torch
from torch.utils import data
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from models.preprocessing import *

# Code for this dataloader is heavily borrowed from PECNet.
# https://github.com/HarshayuGirase/Human-Path-Prediction


def initial_pos_func(traj_batches):
	batches = []
	for b in traj_batches:
		starting_pos = b[:, 7, :].copy()/1000 #starting pos is end of past, start of future. scaled down.
		batches.append(starting_pos)

	return batches


def initial_pos_from_past(traj_batches, past_len):
	batches = []
	past_index = max(past_len - 1, 0)
	for batch in traj_batches:
		starting_pos = batch[:, past_index, :].copy()
		batches.append(starting_pos)
	return batches


def build_seq_start_end_from_masks(masks):
	seq_start_end_list = []
	for m in masks:
		total_num = m.shape[0]
		scene_start_idx = 0
		num_list = []
		for i in range(total_num):
			if i < scene_start_idx:
				continue
			scene_actor_num = np.sum(m[i])
			scene_start_idx += scene_actor_num
			num_list.append(scene_actor_num)
		cum_start_idx = [0] + np.cumsum(np.array(num_list)).tolist()
		seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
		seq_start_end_list.append(seq_start_end)
	return seq_start_end_list


class SocialDataset(data.Dataset):

	def __init__(self, folder, scene, set_name="train", b_size=4096, t_tresh=60, d_tresh=50, homo_matrix=None, verbose=False):
		'Initialization'
		load_name = "./{0}/social_{1}_{2}_{3}_{4}_{5}.pickle".format(folder, scene, set_name, b_size, t_tresh, d_tresh)

		# print(load_name)
		with open(load_name, 'rb') as f:
			data = pickle.load(f)

		traj, masks = data
		traj_new = []

		for t in traj:
			t = np.array(t)
			t = t[:, :, 2:4]
			traj_new.append(t)
			if set_name == "train":
				#augment training set with reversed tracklets...
				# reverse_t = np.flip(t, axis=1).copy()
				ks = [1, 2, 3]
				for k in ks:
					data_rot = rot(t, k).copy()
					traj_new.append(data_rot)
				data_flip = fliplr(t).copy()
				traj_new.append(data_flip)

		masks_new = []
		for m in masks:
			masks_new.append(m)

			if set_name == "train":
				#add second time for the reversed tracklets...
				masks_new.append(m)
				for _ in range(3):
					masks_new.append(m)

		seq_start_end_list = []
		for m in masks:
			total_num = m.shape[0]
			scene_start_idx = 0
			num_list = []
			for i in range(total_num):
				if i < scene_start_idx:
					continue
				scene_actor_num = np.sum(m[i])
				scene_start_idx += scene_actor_num
				num_list.append(scene_actor_num)

			cum_start_idx = [0] + np.cumsum(np.array(num_list)).tolist()
			seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
			seq_start_end_list.append(seq_start_end)
			if set_name == "train":
				#add second time for the reversed tracklets...
				seq_start_end_list.append(seq_start_end)
				for _ in range(3):
					seq_start_end_list.append(seq_start_end)

		# print(len(traj_new), len(seq_start_end_list), len(masks_new))
		traj_new = np.array(traj_new)
		masks_new = np.array(masks_new)

		self.trajectory_batches = traj_new.copy()
		self.mask_batches = masks_new.copy()
		self.initial_pos_batches = np.array(initial_pos_func(self.trajectory_batches)) #for relative positioning
		self.seq_start_end_batches = seq_start_end_list
		if verbose:
			print("Initialized social dataloader...")

	def __len__(self):
		return len(self.trajectory_batches)

	def __getitem__(self, idx):
		trajectory = self.trajectory_batches[idx]
		mask = self.mask_batches[idx]
		initial_pos = self.initial_pos_batches[idx]
		seq_start_end = self.seq_start_end_batches[idx]
		return np.array(trajectory), np.array(mask), np.array(initial_pos), np.array(seq_start_end), None


class JAADPIEDataset(data.Dataset):
	def __init__(self, root_dir, dataset_name, set_name="train", past_len=8, verbose=False):
		self.dataset_name = dataset_name
		self.set_name = set_name
		self.past_len = past_len

		pkl_path = os.path.join(root_dir, f"{set_name}.pkl")
		npz_path = os.path.join(root_dir, f"{set_name}.npz")

		if os.path.exists(pkl_path):
			with open(pkl_path, "rb") as f:
				payload = pickle.load(f)
		elif os.path.exists(npz_path):
			payload = np.load(npz_path, allow_pickle=True)
		else:
			raise FileNotFoundError(
				f"Could not find JAAD/PIE data for split '{set_name}'. "
				f"Expected {pkl_path} or {npz_path}."
			)

		trajectories, masks, seq_start_end, initial_pos, maps = self._unpack_payload(payload)

		self.trajectory_batches = np.asarray(trajectories)
		self.mask_batches = np.asarray(masks)
		self.map_batches = np.asarray(maps) if maps is not None else None

		if seq_start_end is None:
			seq_start_end = build_seq_start_end_from_masks(self.mask_batches)
		self.seq_start_end_batches = seq_start_end

		if initial_pos is None:
			initial_pos = initial_pos_from_past(self.trajectory_batches, self.past_len)
		self.initial_pos_batches = np.asarray(initial_pos)

		if verbose:
			print(f"Initialized {dataset_name.upper()} dataloader with {len(self.trajectory_batches)} samples.")

	def _unpack_payload(self, payload):
		trajectories = masks = seq_start_end = initial_pos = maps = None
		if isinstance(payload, dict):
			trajectories = payload.get("trajectories")
			masks = payload.get("masks")
			seq_start_end = payload.get("seq_start_end")
			initial_pos = payload.get("initial_pos")
			maps = payload.get("maps") or payload.get("semantic_maps")
		elif isinstance(payload, (list, tuple)):
			if len(payload) >= 2:
				trajectories, masks = payload[0], payload[1]
			if len(payload) >= 3:
				seq_start_end = payload[2]
			if len(payload) >= 4:
				initial_pos = payload[3]
			if len(payload) >= 5:
				maps = payload[4]
		else:
			raise ValueError("Unsupported JAAD/PIE payload format. Use dict or tuple/list.")

		if trajectories is None or masks is None:
			raise ValueError("JAAD/PIE payload must include trajectories and masks.")

		return trajectories, masks, seq_start_end, initial_pos, maps

	def __len__(self):
		return len(self.trajectory_batches)

	def __getitem__(self, idx):
		trajectory = self.trajectory_batches[idx]
		mask = self.mask_batches[idx]
		initial_pos = self.initial_pos_batches[idx]
		seq_start_end = self.seq_start_end_batches[idx]
		map_tensor = None if self.map_batches is None else self.map_batches[idx]
		return np.array(trajectory), np.array(mask), np.array(initial_pos), np.array(seq_start_end), map_tensor


def socialtraj_collate(batch):
	trajectories = []
	mask = []
	initial_pos = []
	seq_start_end = []
	maps = []
	has_map = False
	for _batch in batch:
		trajectories.append(_batch[0])
		mask.append(_batch[1])
		initial_pos.append(_batch[2])
		seq_start_end.append(_batch[3])
		map_tensor = _batch[4] if len(_batch) > 4 else None
		if map_tensor is not None:
			has_map = True
		maps.append(map_tensor)

	if has_map:
		if any(m is None for m in maps):
			raise ValueError("Semantic map tensors are missing for some samples in the batch.")
		map_tensor = torch.Tensor(maps).squeeze(0)
	else:
		map_tensor = None

	return (
		torch.Tensor(trajectories).squeeze(0),
		torch.Tensor(mask).squeeze(0),
		torch.Tensor(initial_pos).squeeze(0),
		torch.tensor(seq_start_end, dtype=torch.int32).squeeze(0),
		map_tensor,
	)
