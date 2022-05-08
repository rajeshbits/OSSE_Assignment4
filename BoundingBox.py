#!usr/bin/env python

import os
import sys
import glob
import random
import numpy as np
import time
import shutil
import torch

# adding Carla .egg file to system path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def main():

	imagepath = "D:\\Carla_Compiled\\CARLA_0.9.9\\WindowsNoEditor\\PythonAPI\\examples\\BoundingBoxPred"

	def clean():
		if os.path.exists(imagepath):											# Cleaning previous images
			for folders in glob.glob(imagepath + '//*'):
				try:
					shutil.rmtree(folders)
				except:					
					break
	

	class CarEnv():
		actor_list = []
		
		def __init__(self):
			
			#for f in glob.glob(imagepath + '//*'):
			#	try:
			#		os.remove(f)
			#	except:
			#		break

			self.client = carla.Client('localhost',2000)
			self.client.set_timeout(5)
			self.World = self.client.get_world()
			self.BlueprintLibrary = self.World.get_blueprint_library()
			self.FV_bp = self.BlueprintLibrary.filter('audi')[0]
			self.RV_bp = self.BlueprintLibrary.filter('model3')[0]


		def create_dir(self, imagepath,episode,x,y):

			if os.path.exists(imagepath + f'\\{episode}'):
				pass
			else:
				os.mkdir(imagepath + f'\\{episode}_{int(x)}_{y}')

		def transform(self,sp_point,x,y, episode,actor_list):
			self.Transform = self.World.get_map().get_spawn_points()[sp_point]
			self.RV = self.World.spawn_actor(self.RV_bp,self.Transform)
			actor_list.append(self.RV)
			self.Transform.location.x += x
			self.Transform.location.y += y
			try:
				self.FV = self.World.spawn_actor(self.FV_bp,self.Transform)
				actor_list.append(self.FV)
			except:
				print(f'failed to spawn due to collision at spawn point:{sp_point} with +x :{x}  & + y :{y}!')
				flag = 1
				return flag
	
			self.Cam = self.BlueprintLibrary.find('sensor.camera.rgb')
			self.Cam_Transform = carla.Transform(carla.Location(x=2, z=2))
			self.Camera = self.World.spawn_actor(self.Cam, self.Cam_Transform, attach_to = self.RV,attachment_type = carla.AttachmentType.Rigid)
			image_name = self.Calc_State_Tensor()
			self.Camera.listen(lambda image: image.save_to_disk(f'BoundingBoxPred/{episode}_{sp_point}/{image_name}_%.6d.png' % image.frame))
			time.sleep(1)
			actor_list.append(self.Camera)

		def Calc_State_Tensor(self):
			fv_arr_x,fv_arr_y,fv_arr_z,rr_arr_x,rr_arr_y,rr_arr_z = np.zeros(shape=(1,1)), np.zeros(shape=(1,1)), np.zeros(shape=(1,1)),np.zeros(shape=(1,1)), np.zeros(shape=(1,1)), np.zeros(shape=(1,1)) # Initialize arrays for appending
			for point in range(8): # Range from 0 to 8 i.e., 9 to accomodate first rows of zeros.
				# For Front Vehicle
				fv_arr_x = np.vstack((fv_arr_x,self.FV.bounding_box.get_world_vertices(self.FV.get_transform())[point].x)) # Vertical stacking x coordinates for all points
				fv_arr_y = np.vstack((fv_arr_y,self.FV.bounding_box.get_world_vertices(self.FV.get_transform())[point].y)) # Vertical stacking y coordinates for all points
				fv_arr_z = np.vstack((fv_arr_z,self.FV.bounding_box.get_world_vertices(self.FV.get_transform())[point].z)) # Vertical stacking z coordinates for all points
				# For Rear Vehicle
				rr_arr_x = np.vstack((rr_arr_x,self.RV.bounding_box.get_world_vertices(self.RV.get_transform())[point].x)) # Vertical stacking x coordinates for all points
				rr_arr_y = np.vstack((rr_arr_y,self.RV.bounding_box.get_world_vertices(self.RV.get_transform())[point].y)) # Vertical stacking y coordinates for all points
				rr_arr_z = np.vstack((rr_arr_z,self.RV.bounding_box.get_world_vertices(self.RV.get_transform())[point].z)) # Vertical stacking z coordinates for all points

			DiffArr = np.hstack((fv_arr_x[1:,:].transpose(),fv_arr_y[1:,:].transpose(),fv_arr_z[1:,:].transpose())) - np.hstack((rr_arr_x[1:,:].transpose(),rr_arr_y[1:,:].transpose(),rr_arr_z[1:,:].transpose()))
			rounder = lambda x:str(round(x,2))
			DiffArr_str = [rounder(x) for x in DiffArr.flatten()]
			img_name = ''
			img_name += '_'.join(DiffArr_str)
			
			return img_name
		

		def destroy(self,actor_list):
			for actor in actor_list:
				actor.destroy()
			print('Actors destroyed !!')



	episodes = 3
	spawn_points = 30

	clean()
	for episode in range(episodes):
		for spawn_point in range(20,spawn_points):
			actor_list = []
			carenv = CarEnv()

			x = np.random.choice(np.linspace(3,20,5))
			y = np.random.choice(np.arange(-0.2,0.2,0.05))

			#carenv.create_dir(imagepath,episode,x,y)
			flag = carenv.transform(spawn_point,x,y,episode,actor_list)
			if flag == 1:
				carenv.destroy(actor_list)
				continue
			carenv.Calc_State_Tensor()
			carenv.destroy(actor_list)
			if episode == episodes -1:
				time.sleep(1)

if __name__ == '__main__':
	main()