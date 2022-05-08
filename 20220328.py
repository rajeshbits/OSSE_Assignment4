#!/usr/bin/env python

# imports
import sys
import glob
import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import math



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

	# Initializing empty lists
	#Actor_list = []
	#Collision_hist = []
	#Loss_hist = []
	#Reward_hist = []
	#Action_hist = []
	#Intensity_hist = []
	#Lane_Invasion_hist = []
	num_episodes = 15

	imagepath = "D:\\Carla_Compiled\\CARLA_0.9.9\\WindowsNoEditor\\PythonAPI\\examples\\_out"
	if os.path.exists(imagepath):																	# Cleaning previous images
		for folders in glob.glob(imagepath + '//*'):
			shutil.rmtree(folders)	
																
	class CarEnv:
		def __init__(self,episode,epsilon,cam,pretrained=False):
			self.client = carla.Client('localhost',2000)								# Create carla client
			self.client.set_timeout(5) 													# 5 seconds timeout set to check network connectivity issues
			self.World = self.client.get_world()										# world from client
			self.Blueprint_Library = self.World.get_blueprint_library()					# Blueprint library from world
			self.FV_bp = self.Blueprint_Library.filter('audi')[0]						# Front vehicle Audi from Blueprint library
			self.RV_bp = self.Blueprint_Library.filter('model3')[0]						# Rear vehicle Tesla from Blueprint library

			self.Transform = self.World.get_map().get_spawn_points()[1]					# Location of Front Vehicle
			self.RV = self.World.spawn_actor(self.RV_bp,self.Transform)					# Spawning Front Vehicle
			Actor_list.append(self.RV)

			self.Transform.location.x += 15												# Location of Rear Vehicle
			self.FV = self.World.spawn_actor(self.FV_bp,self.Transform)					# Spawning Rear Vehicle
			Actor_list.append(self.FV)

			imagepath = "D:\\Carla_Compiled\\CARLA_0.9.9\\WindowsNoEditor\\PythonAPI\\examples\\_out"		# Creating camera image saving path

			self.model_path = "D:\\Carla_Compiled\\CARLA_0.9.9\\WindowsNoEditor\\PythonAPI\\examples\\model"		# Creating model saving path
			try:
				os.mkdir(self.model_path)
				print('model folder created!')
			except:
				pass

			self.Cam = self.Blueprint_Library.find('sensor.camera.rgb')																				# finding camera sensor
			self.Cam_Transform = carla.Transform(carla.Location(x=-10, z=5))																		# Camera transform
			self.Camera = self.World.spawn_actor(self.Cam, self.Cam_Transform, attach_to = self.RV,attachment_type = carla.AttachmentType.Rigid)	# Spawning camera
			if cam == True:
				os.mkdir(imagepath + f"\\{episode}")
				self.Camera.listen(lambda image: image.save_to_disk(f'_out/{episode}/%.6d.png' % image.frame))														# saving images to disk
			Actor_list.append(self.Camera)
			time.sleep(1)																															# waiting for camera to start

			self.RV.set_velocity(carla.Vector3D(60*10/36, 0.0, 0.0))						# set velocity to rear vehicle (50kmph)
			self.episode_start_time = time.time() 											# time from which rear vehicle started to move

			self.best_action_count = 0
			self.random_action_count = 0

			# Creating collision sensor
			self.Col = self.Blueprint_Library.find('sensor.other.collision')														# finding collision sensor
			self.Col_Transform = self.Cam_Transform																					# Collision sensor transform same as camera
			self.Col_sensor = self.World.spawn_actor(self.Col, self.Col_Transform, attach_to = self.RV, attachment_type = carla.AttachmentType.Rigid) # Spawning collision sensor
			self.Col_sensor.listen(lambda event : self.collision_handler(event))														# recording collision
			Actor_list.append(self.Col_sensor)
			self.Col_Flag = 0

			# Creating Lane Invasion Sensor
			self.Lane_Invasion = self.Blueprint_Library.find('sensor.other.lane_invasion')
			self.Lane_Invasion_Transform = self.Cam_Transform
			self.Lane_Invasion_Sensor = self.World.spawn_actor(self.Lane_Invasion,self.Lane_Invasion_Transform, attach_to=self.RV, attachment_type = carla.AttachmentType.Rigid)
			self.Lane_Invasion_Sensor.listen(lambda event: self.lane_callback(event))

		def lane_callback(self,event):
			Lane_Invasion_hist.append(event.crossed_lane_markings)
			#print("Lane invasion detected:\n"+str(event)+'\n')


		def collision_handler(self,event):
			other_actor = event.other_actor
			impulse = event.normal_impulse
			intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
			Collision_hist.append(event)
			Intensity_hist.append(intensity)
			#print("Collision detected:\n"+str(event)+'\n')
			#print(f'Collision_hist: {Collision_hist}')

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
			return torch.Tensor(DiffArr.flatten())
	
		def create_DQN(self):
			self.model = torch.nn.Sequential(									# Creating sequential pytorch neural network
										nn.Linear(24,50),
										nn.ReLU(),
										nn.Linear(50,50),
										nn.ReLU(),
										nn.Linear(50,6))

			self.criterion = nn.MSELoss()
			self.optimizer = Adam(self.model.parameters(),lr = 0.01)
			return self.model,self.criterion,self.optimizer


		def choose_action(self,action):											# Steer/brake/throttle done by agent on choosing action
		# Select action
			if action == 0:
				self.RV.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
			if action == 1:
				self.RV.apply_control(carla.VehicleControl(throttle=1.0, steer=-1))
			if action == 2:
				self.RV.apply_control(carla.VehicleControl(throttle=1.0, steer=1))
			if action == 3:
				self.RV.apply_control(carla.VehicleControl(throttle=0.0, steer=0, brake = 1))
			if action == 4:
				self.RV.apply_control(carla.VehicleControl(throttle=0.0, steer=-1,brake=1))
			if action == 5:
				self.RV.apply_control(carla.VehicleControl(throttle=0.0, steer=1,brake = 1))

		def Check_Col_Flag(self):
			if len(Collision_hist) != 0:
				Col_Flag = 1
			else:
				Col_Flag = 0
			return Col_Flag

		def Check_Lane_Inv_Flag(self):
			if len(Lane_Invasion_hist) != 0:
				Lane_Inv_Flag = 1
			else:
				Lane_Inv_Flag = 0
			return Lane_Inv_Flag

		def calc_reward(self,action):												# update reward function for vehicle moving from a to b
			loss_calc_time_interval = time.time() - self.episode_start_time
			RV_loc = self.FV.bounding_box.get_world_vertices(self.FV.get_transform())[0].x 
			X_Destination = 120
			#print(f'RV position:{X_Destination - RV_loc}')
			self.Col_Flag = self.Check_Col_Flag()
			self.Lane_Inv_Flag = self.Check_Lane_Inv_Flag()
			vel = self.RV.get_velocity()
			vel = int(3.6 * vel.x)
			if self.Col_Flag == 0:
				if self.Lane_Inv_Flag == 0:
					self.reward = (X_Destination - RV_loc)*(-1)
					#if vel < 5:
					#	self.reward -= 10
					#if vel > 5:
					#	self.reward += 5
				else:
					self.reward = (X_Destination - RV_loc)*(-1)
					self.reward -= 0
					#if vel < 5:
					#	self.reward -= 10
					#if vel > 5:
					#	self.reward += 10
			elif self.Col_Flag == 1:
				if self.Lane_Inv_Flag == 0:
					self.reward = (X_Destination - RV_loc)*(-1)
					self.reward -= 500
					self.reward -= Intensity_hist[-1]
					#if vel < 5:
					#	self.reward -= 10
					#if vel > 5:
					#	self.reward += 10
				else:
					self.reward = (X_Destination - RV_loc)*(-1)
					self.reward -= 500
					self.reward -= Intensity_hist[-1]
					self.reward -= 0
					#if vel < 5:
					#	self.reward -= 100
					#if vel > 5:
					#	self.reward += 100
			return self.reward


		def train(self,episode,pretrained, save_model_path=None,pretrained_model_path=None):
			load_count = 0
			pretrained_load_count = 0
			while ((len(Collision_hist) == 0) or ((time.time() - self.episode_start_time) < 3)):
				if episode == 0 and pretrained == False:
					self.model,criterion,optimizer = self.create_DQN()
				elif episode == 0 and pretrained == True:
					self.model,criterion,optimizer = self.load_model(pretrained_model_path)
					pretrained_load_count = 1
					pretrained = False
				else:
					if load_count == 0 and pretrained == False:
						self.model,criterion,optimizer = self.load_model(save_model_path)
						load_count = 1
					elif load_count == 0 and pretrained == True and pretrained_load_count == 1:
						save_model_path = pretrained_model_path
						self.model,criterion,optimizer = self.load_model(save_model_path)
				output = self.model(self.Calc_State_Tensor())
				soft = nn.Softmax(dim=0)
				#print(soft(output))
				prob = soft(output) / torch.sum(soft(output))

				#if np.random.randn() > epsilon:							# epsilon greedy approach for exploration and exploitation
				#	action = torch.argmax(soft(output)).item()
				#	self.choose_action(action)
				#	self.best_action_count += 1
				#	Best_Action_hist.append(action)
				#else:
				#	action = np.random.choice([-3,-2,-1,1,2,3])
				#	self.choose_action(action)
				#	self.random_action_count += 1
				#	Random_Action_hist.append(action)

				action = np.random.choice([-3,-2,-1,1,2,3], p = prob.detach().numpy())
				Selected_Action_hist.append(action)
				target = output.detach().clone()			# Cloning target as DQN output
				target[action] = self.calc_reward(action) 		# Updating action reward in target vector for current action index

				mseloss = nn.MSELoss()						# Intializing error function
				optimizer.zero_grad()						# zeroing optimizer gradients
				loss = mseloss(output,target)				# Calculating loss
				loss.backward()								# Backpropogating gradients
				optimizer.step()							# Optimizing parameters

				Loss_hist.append(loss.item())
				Reward_hist.append(self.calc_reward(action))
				Action_hist.append(action)
				Action_Reward_hist.append((action,self.calc_reward(action)))


		def save_model(self):
			save_model_path = self.model_path + f'\\__Episode#{episode}__Col{self.Check_Col_Flag()}__LaneInv{self.Check_Lane_Inv_Flag()}__AvgReward{np.mean(Reward_hist):_>7.2f}.model'
			torch.save(self.model.state_dict(),save_model_path)
			print(f'Saving model at {save_model_path}')
			return save_model_path

		def load_model(self,save_model_path):
			self.model,criterion,optimizer = self.create_DQN()
			print(f'Loading model from {save_model_path}')
			self.model.load_state_dict(torch.load(save_model_path))
			return self.model,criterion,optimizer

	Avg_Loss = []
	epsilon = 1
	EPSILON_DECAY = 0.9975 ## 0.9975 99975
	MIN_EPSILON = 0.001

	for episode in range(num_episodes):
		Actor_list = []
		Collision_hist = []
		Loss_hist = []
		Reward_hist = []
		Action_hist = []
		Intensity_hist = []
		Lane_Invasion_hist = []
		Best_Action_hist = []
		Random_Action_hist = []
		Selected_Action_hist = []
		Action_Reward_hist = []

		pretrained = True

		if episode == 0 and pretrained == True:
			pretrained = True
		else:
			pretrained = False

		if ((episode % 5 == 0) or (episode == num_episodes-1)):
			cam = True
		else:
			cam = False 

		# Decay epsilon
		if epsilon > MIN_EPSILON:
			epsilon *= EPSILON_DECAY
			epsilon = max(MIN_EPSILON, epsilon)

		model_path = "D:\\Carla_Compiled\\CARLA_0.9.9\\WindowsNoEditor\\PythonAPI\\examples\\model"
		pretrained_model_path = "D:\\Carla_Compiled\\CARLA_0.9.9\\WindowsNoEditor\\PythonAPI\\examples\\latest_model"

		if pretrained == True:
			pretrained_model_path = glob.glob(pretrained_model_path + '/*')[0]
			save_model_path = None
			target = np.zeros(6)

		if episode == 0 and pretrained == False:
			save_model_path = None
			files = glob.glob(model_path + '/*')
			for file in files:
				os.remove(file)
		else:
			pass

		#save_model_path = model_path
		env = CarEnv(episode,epsilon,cam,pretrained)
		print('*'*25)
		print(f'Training for episode # {episode}..')								# Initializing environment
		print('*'*25)
		env.train(episode, pretrained, save_model_path, pretrained_model_path)
		save_model_path = env.save_model()
		for actor in Actor_list:
			actor.destroy()
		print('\nEpisode Over! Actors Destroyed!!\n')
		Avg_Loss.append(np.mean(Loss_hist))
		episode_duration = time.time() - env.episode_start_time
		#print(f'Episode duration : {episode_duration}')
		#print(f'Best action count:{env.best_action_count}, Random_action count:{env.random_action_count}')
		#print(f'best action hist:{Best_Action_hist}, \nrandom action hist: {Random_Action_hist},\nSelected action list: {Selected_Action_hist}\nAction_Reward_hist:{Action_Reward_hist}')
	

	fig,ax = plt.subplots(1,1)
	print(f'Lane_Invasion_history: {Lane_Invasion_hist}')
	ax.plot(range(len(Avg_Loss)),Avg_Loss)
	ax.set_xlabel('Timesteps')
	ax.set_ylabel('Avg Loss')
	#plt.show()


if __name__ == '__main__':
	main()







