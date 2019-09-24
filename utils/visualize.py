import os
import sys
import io

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, axis3d, proj3d
from PIL import Image
from graphviz import Digraph

import torch
from torch.autograd import Variable


def make_dot(var, params=None):
	""" Produces Graphviz representation of PyTorch autograd graph

	Blue nodes are the Variables that require grad, orange are Tensors
	saved for backward in torch.autograd.Function

	Args:
		var: output Variable
		params: dict of (name, Variable) to add names to node that
			require grad (TODO: make optional)
	"""
	if params is not None:
		assert all(isinstance(p, Variable) for p in params.values())
		param_map = {id(v): k for k, v in params.items()}

	node_attr = dict(style='filled',
					 shape='box',
					 align='left',
					 fontsize='12',
					 ranksep='0.1',
					 height='0.2')
	dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
	seen = set()

	def size_to_str(size):
		return '('+(', ').join(['%d' % v for v in size])+')'

	def add_nodes(var):
		if var not in seen:
			if torch.is_tensor(var):
				dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
			elif hasattr(var, 'variable'):
				u = var.variable
				name = param_map[id(u)] if params is not None else ''
				node_name = '%s\n %s' % (name, size_to_str(u.size()))
				dot.node(str(id(var)), node_name, fillcolor='lightblue')
			else:
				dot.node(str(id(var)), str(type(var).__name__))
			seen.add(var)
			if hasattr(var, 'next_functions'):
				for u in var.next_functions:
					if u[0] is not None:
						dot.edge(str(id(u[0])), str(id(var)))
						add_nodes(u[0])
			if hasattr(var, 'saved_tensors'):
				for t in var.saved_tensors:
					dot.edge(str(id(t)), str(id(var)))
					add_nodes(t)
	add_nodes(var.grad_fn)
	return dot


def pts2img(pts, clr=None):
	def crop_image(image):
		image_data = np.asarray(image)
		assert (len(image_data.shape) == 3)
		image_data_bw = image_data.max(axis=2) if image_data.shape[-1] <= 3 else image_data[:, :, 3]
		non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
		non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
		cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
		image_data_new = image_data[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1, :]
		# return Image.fromarray(image_data_new)
		return image_data_new
	plt.close('all')
	fig = plt.figure()
	fig.set_rasterized(True)
	ax = axes3d.Axes3D(fig)
	pts -= np.mean(pts,axis=0) #demean

	ax.view_init(20,110) # (20,30) for motor, (20, 20) for car, (20,210), (180, 60) for lamp, ax.view_init(20,110) for general, (180,60) for lamp, (20, 200) for mug and pistol
	ax.set_alpha(255)
	ax.set_aspect('auto')
	min_lim = pts.min()
	max_lim = pts.max()
	ax.set_xlim3d(min_lim,max_lim)
	ax.set_ylim3d(min_lim,max_lim)
	ax.set_zlim3d(min_lim,max_lim)

	ax.scatter(
		pts[:, 0], pts[:, 1], pts[:, 2],
		c=clr,
		zdir='x',
		s=20,
		edgecolors=(0.5, 0.5, 0.5)  # (0.5,0.5,0.5)
	)

	ax.set_axis_off()
	ax.set_facecolor((1,1,1,0))
	ax.set_rasterized(True)
	ax.set_rasterization_zorder(1)
	ax.set_facecolor("white")
	buf = io.BytesIO()
	plt.savefig(
		buf, format='png', transparent=True,
		bbox_inches='tight', pad_inches=0,
		rasterized=True,
		dpi=200
	)
	buf.seek(0)
	im = Image.open(buf)
	im_data = np.asarray(im)
	# im = crop_image(im)
	buf.close()
	return im_data