#!/usr/bin/env python

# imports
import sys
import glob
import os
import time
import shutil
import numpy as np



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
	actor_list = []															# Creating empty actor list

	Carla_client = carla.Client('localhost',2000)							# Create carla client
	Carla_client.set_timeout(5) 											# 5 seconds timeout set to check network connectivity issues
	World = Carla_client.get_world()										# world from client
	Blueprint_Library = World.get_blueprint_library()						# Blueprint library from world
	FV_bp = Blueprint_Library.filter('audi')[0]								# Front vehicle Audi from Blueprint library
	RV_bp = Blueprint_Library.filter('model3')[0]							# Rear vehicle Tesla from Blueprint library

	Transform = World.get_map().get_spawn_points()[1]						# Location of Front Vehicle
	RV = World.spawn_actor(RV_bp,Transform)									# Spawning Front Vehicle
	actor_list.append(RV)

	Transform.location.x += 25												# Location of Rear Vehicle
	FV = World.spawn_actor(FV_bp,Transform)									# Spawning Rear Vehicle
	actor_list.append(FV)


	
	imagepath = "D:\\Carla_Compiled\\CARLA_0.9.9\\WindowsNoEditor\\PythonAPI\\examples\\_out"		# Creating camera image saving path
	if os.path.exists(imagepath):																	# Cleaning previous images
		shutil.rmtree(imagepath)

	Cam = Blueprint_Library.find('sensor.camera.rgb')												# finding camera blueprint
	Cam_Transform = carla.Transform(carla.Location(x=0, z=2))
	Camera = World.spawn_actor(Cam, Cam_Transform, attach_to = RV,attachment_type = carla.AttachmentType.Rigid)
	Camera.listen(lambda image: image.save_to_disk('_out/%.6d.png' % image.frame))
	print(f'{Camera.type_id} created! ')
	actor_list.append(Camera)

	#time.sleep(5)

	#print(RV.bounding_box.get_world_vertices(RV.get_transform())[0].x)
	#print(RV.bounding_box.get_world_vertices(RV.get_transform())[0].y)
	#print(RV.bounding_box.get_world_vertices(RV.get_transform())[0].z)

	fv_arr_x,fv_arr_y,fv_arr_z,rr_arr_x,rr_arr_y,rr_arr_z = np.zeros(shape=(1,1)), np.zeros(shape=(1,1)), np.zeros(shape=(1,1)),np.zeros(shape=(1,1)), np.zeros(shape=(1,1)), np.zeros(shape=(1,1)) # Initialize arrays for appending
	for point in range(8): # Range from 0 to 8 i.e., 9 to accomodate first rows of zeros.
		# For Front Vehicle
		fv_arr_x = np.vstack((fv_arr_x,FV.bounding_box.get_world_vertices(FV.get_transform())[point].x)) # Vertical stacking x coordinates for all points
		fv_arr_y = np.vstack((fv_arr_y,FV.bounding_box.get_world_vertices(FV.get_transform())[point].y)) # Vertical stacking y coordinates for all points
		fv_arr_z = np.vstack((fv_arr_z,FV.bounding_box.get_world_vertices(FV.get_transform())[point].z)) # Vertical stacking z coordinates for all points
		# For Rear Vehicle
		rr_arr_x = np.vstack((rr_arr_x,RV.bounding_box.get_world_vertices(RV.get_transform())[point].x)) # Vertical stacking x coordinates for all points
		rr_arr_y = np.vstack((rr_arr_y,RV.bounding_box.get_world_vertices(RV.get_transform())[point].y)) # Vertical stacking y coordinates for all points
		rr_arr_z = np.vstack((rr_arr_z,RV.bounding_box.get_world_vertices(RV.get_transform())[point].z)) # Vertical stacking z coordinates for all points

	DiffArr = np.hstack((fv_arr_x[1:,:].transpose(),fv_arr_y[1:,:].transpose(),fv_arr_z[1:,:].transpose())) - np.hstack((rr_arr_x[1:,:].transpose(),rr_arr_y[1:,:].transpose(),rr_arr_z[1:,:].transpose()))
	print(f'DiffArr shape: {DiffArr.shape}')
	print(DiffArr.flatten())




	for actor in actor_list:
		actor.destroy()
	print('done')


if __name__ == '__main__':
	main()







