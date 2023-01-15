#!/usr/bin/python

import os, logging

from datetime import datetime

import tensorflow as tf
from struct import pack, unpack

import numpy as np
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K
import random, math
from src import config

def bitflip(f, pos):
	
	""" Single bit-flip in 32 bit floats """

	f_ = pack('f', f)
	b = list(unpack('BBBB', f_))
	[q, r] = divmod(pos, 8)
	b[q] ^= 1 << r
	f_ = pack('BBBB', *b)
	f = unpack('f', f_)
	return f[0]

class inject():
	def __init__(
		self, model, confFile, log_level="DEBUG", **kwargs
		):

		# Logging setup
		log_dir = os.path.join(os.environ['CONDA_PREFIX'],"TensorFI2/logs")
		if not os.path.exists(log_dir):
			os.mkdir(log_dir)

		self.logger = open(os.path.join(log_dir, datetime.now().strftime('TFI2_%d_%m_%Y_%H_%M_%S.log')), "w")
  
		#logging.basicConfig(filename=os.path.join(log_dir, datetime.now().strftime('TFI2_%d_%m_%Y_%H_%M_%S.log')), format="%(message)s", filemode='w')

		# self.logger = logging.getLogger()
		# self.logger.setLevel(log_level)
		# self.logger.debug("Logging level set to {0}".format(log_level) + "\n")

		# Retrieve config params
		fiConf = config.config(confFile)
		self.Model = model # No more passing or using a session variable in TF v2

		# Call the corresponding FI function
		fiFunc = getattr(self, fiConf["Target"])
		fiFunc(model, fiConf, **kwargs)

	def layer_states(self, model, fiConf, **kwargs):
		
		""" FI in layer states """
		
		if(fiConf["Mode"] == "single"):

			""" Single layer fault injection mode """
			
			self.logger.write(str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + "\n")
			self.logger.write("----------------Starting fault injection in a random layer----------------\n\n")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			self.logger.write("Type of Fault: " + str(fiConf["Type"]) + "\n")
			self.logger.write("Amount of Faults: " + str(fiConf["Amount"]) + "\n")

			# Choose a random layer for injection
			randnum = random.randint(0, len(model.trainable_variables) - 1)

			# Get layer states info
			v = model.trainable_variables[randnum]
			num = v.shape.num_elements()

			self.logger.write("\nOriginal Layer [" + str(randnum) + "]: \n")
			self.logger.write(str(model.trainable_variables[randnum].numpy()) + "\n")
			self.logger.write("\nNumber of Elements in Layer: " + str(v.shape.num_elements()) + "\n")

			if(fiFault == "zeros"):
				fiSz = (fiSz * num) / 100
				fiSz = math.floor(fiSz)

			# Choose the indices for FI
			#ind = random.sample(range(num), fiSz)
			ind = np.random.choice(range(num), fiSz, replace=True)

			self.logger.write("Layer Element Indicies (#s) to Inject: " + str(ind) + "\n")

			# Unstack elements into a single dimension
			elem_shape = v.shape
			v_ = tf.identity(v)
			v_ = tf.keras.backend.flatten(v_)
			v_ = tf.unstack(v_)

			# Inject the specified fault into the randomly chosen values
			item_counter = 0

			if(fiFault == "zeros"):
				for item in ind:
					self.logger.write("("+ str(item_counter) +") Original Element #" + str(item) + " Value: " + str(float(v_[item])) + "\n")
					v_[item] = 0.
					item_counter+=1
			elif(fiFault == "random"):
				for item in ind:
					self.logger.write("("+ str(item_counter) +") Original Element #" + str(item) + " Value: " + str(float(v_[item])) + "\n")
					v_[item] = np.random.random()
					self.logger.write("("+ str(item_counter) +") Faulty Element #" + str(item) + " Value:   " + str(v_[item]) + "\n")
					item_counter+=1
			elif(fiFault == "bitflips"):
				for item in ind:
					val = v_[item]
					
					# If random bit chosen to be flipped
					if(fiConf["Bit"] == "N"):
						pos = random.randint(0, 31)
						self.logger.write("("+ str(item_counter) +") Bit Position in 32-Bit Element #" + str(item) + " to be Flipped: " + str(pos) + "\n")

					# If bit position specified for flip
					else:
						pos = int(fiConf["Bit"])
						self.logger.write("("+ str(item_counter) +") Bit Position in 32-Bit Element #" + str(item) + " to be Flipped:" + str(pos) + "\n")

					val_ = bitflip(val, pos)
					self.logger.write("("+ str(item_counter) +") Original #" + str(item) + " Value: " + str(float(v_[item])) + "\n")
					self.logger.write("("+ str(item_counter) +") Faulty #" + str(item) + " Value:   " + str(val_) + "\n")
					self.logger.write("\n")
					item_counter+=1
					v_[item] = val_

			# Reshape into original dimensions and store the faulty tensor
			v_ = tf.stack(v_)
			v_ = tf.reshape(v_, elem_shape)
			v.assign(v_)

			self.logger.write("Faulty Layer [" + str(randnum) + "]: \n\n")
			self.logger.write(str(tf.reshape(v_, elem_shape).numpy()) + "\n")

			self.logger.write("\nCompleted injections... exiting")
			self.logger.close()

		elif(fiConf["Mode"] == "multiple"):

			""" Multiple layer fault injection mode """

			self.logger.write(str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))+":" + "\n")
			self.logger.write("----------------Starting fault injection in all layers----------------\n\n")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			self.logger.write("Type of Fault: " + str(fiConf["Type"]) + "\n")
			self.logger.write("Amount of Faults: " + str(fiConf["Amount"]) + "\n")

			# Loop through each available layer in the model
			for n in range(len(model.trainable_variables) - 1):

				# Get layer states info
				v = model.trainable_variables[n]
				num = v.shape.num_elements()

				self.logger.write("\nOriginal Layer [" + str(n) + "]: \n\n")
				self.logger.write(str(model.trainable_variables[n].numpy()) + "\n")
				self.logger.write("\nNumber of Elements in Layer: " + str(v.shape.num_elements()) + "\n")

				if(fiFault == "zeros"):
					fiSz = (fiSz * num) / 100
					fiSz = math.floor(fiSz)

				# Choose the indices for FI
				#ind = random.sample(range(num), fiSz)
				ind = np.random.choice(range(num), fiSz, replace=True)

				self.logger.write("Layer Element Indicies (#s) to Inject: " + str(ind) + "\n\n")

				# Unstack elements into a single dimension
				elem_shape = v.shape
				v_ = tf.identity(v)
				v_ = tf.keras.backend.flatten(v_)
				v_ = tf.unstack(v_)

				# Inject the specified fault into the randomly chosen values
				item_counter = 0

				if(fiFault == "zeros"):
					for item in ind:
						self.logger.write("("+ str(item_counter) +") Original Element #" + str(item) + " Value: " + str(float(v_[item])) + "\n")
						v_[item] = 0.
						item_counter+=1
				elif(fiFault == "random"):
					for item in ind:
						self.logger.write("("+ str(item_counter) +") Original Element #" + str(item) + " Value: " + str(float(v_[item])) + "\n")
						v_[item] = np.random.random()
						self.logger.write("("+ str(item_counter) +") Faulty Element #" + str(item) + " Value:   " + str(v_[item]) + "\n")
						item_counter+=1
				elif(fiFault == "bitflips"):
					for item in ind:
						val = v_[item]

						# If random bit chosen to be flipped
						if(fiConf["Bit"] == "N"):
							pos = random.randint(0, 31)
							self.logger.write("("+ str(item_counter) +") Bit Position in 32-Bit Element #" + str(item) + " to be Flipped: " + str(pos) + "\n")


						# If bit position specified for flip
						else:
							pos = int(fiConf["Bit"])
						val_ = bitflip(val, pos)
						self.logger.write("("+ str(item_counter) +") Original #" + str(item) + " Value: " + str(float(v_[item])) + "\n")
						self.logger.write("("+ str(item_counter) +") Faulty #" + str(item) + " Value:   " + str(val_) + "\n")
						self.logger.write("\n")
						item_counter+=1
						v_[item] = val_

				# Reshape into original dimensions and store the faulty tensor
				v_ = tf.stack(v_)
				v_ = tf.reshape(v_, elem_shape)
				v.assign(v_)

				self.logger.write("Faulty Layer [" + str(n) + "]: \n\n")
				self.logger.write(str(tf.reshape(v_, elem_shape).numpy()) + "\n")

			self.logger.write("\nCompleted injections... exiting")
			self.logger.close()

	def layer_outputs(self, model, fiConf, **kwargs):

		""" FI in layer computations/outputs """

		if(fiConf["Mode"] == "single"):

			""" Single layer fault injection mode """

			self.logger.write(str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))+":" + "\n")
			self.logger.write("----------------Starting fault injection in a random layer----------------\n\n")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			self.logger.write("Type of Fault: " + str(fiConf["Type"]) + "\n")
			self.logger.write("Amount of Faults: " + str(fiConf["Amount"]) + "\n")

			# Get the input for which dynamic injection is to be done
			x_test = kwargs["x_test"]

			# Choose a random layer for injection
			randnum = random.randint(0, len(model.layers) - 2)

			fiLayer = model.layers[randnum]

			# Get the outputs of the chosen layer
			get_output = K.function([model.layers[0].input], [fiLayer.output])

			fiLayerOutputs = get_output([x_test])

			self.logger.write("\n Original Layer [" + str(randnum) + "] Outputs: \n\n")
			self.logger.write(str(get_output([x_test])) + "\n")

			# Unstack elements into a single dimension
			elem_shape = fiLayerOutputs[0].shape
			fiLayerOutputs[0] = fiLayerOutputs[0].flatten()
			num = fiLayerOutputs[0].shape[0]

			if(fiFault == "zeros"):
				fiSz = (fiSz * num) / 100
				fiSz = math.floor(fiSz)

			# Choose the indices for FI
			#ind = random.sample(range(num), fiSz)
			ind = np.random.choice(range(num), fiSz, replace=True)

			self.logger.write("\nLayer Output Indicies (#s) to Inject: " + str(ind) + "\n\n")

			# Inject the specified fault into the randomly chosen values
			item_counter = 0

			if(fiFault == "zeros"):
				for item in ind:
					self.logger.write("("+ str(item_counter) +") Original Output #" + str(item) + " Value: " + str(float(fiLayerOutputs[0][item])) + "\n")
					fiLayerOutputs[0][item] = 0.
					item_counter+=1
			elif(fiFault == "random"):
				for item in ind:
					self.logger.write("("+ str(item_counter) +") Original Output #" + str(item) + " Value: " + str(float(fiLayerOutputs[0][item])) + "\n")
					fiLayerOutputs[0][item] = np.random.random()
					self.logger.write("("+ str(item_counter) +") Faulty Output #" + str(item) + " Value:   " + str(fiLayerOutputs[0][item]) + "\n")
					item_counter+=1
			elif(fiFault == "bitflips"):
				for item in ind:
					val = fiLayerOutputs[0][item]
					if(fiConf["Bit"] == "N"):
						pos = random.randint(0, 31)
						self.logger.write("("+ str(item_counter) +") Bit Position in 32-Bit Output #" + str(item) + " to be Flipped: " + str(pos) + "\n")
					else:
						pos = int(fiConf["Bit"])
						self.logger.write("("+ str(item_counter) +") Bit Position in 32-Bit Output #" + str(item) + " to be Flipped:" + str(pos) + "\n")

					val_ = bitflip(val, pos)
					fiLayerOutputs[0][item] = val_

					self.logger.write("("+ str(item_counter) +") Original #" + str(item) + " Value: " + str(float(val)) + "\n")
					self.logger.write("("+ str(item_counter) +") Faulty #" + str(item) + " Value:   " + str(val_) + "\n")
					self.logger.write("\n")
					item_counter+=1

			# Reshape into original dimensions and get the final prediction
			fiLayerOutputs[0] = fiLayerOutputs[0].reshape(elem_shape)
			get_pred = K.function([model.layers[randnum + 1].input], [model.layers[-1].output])
			pred = get_pred([fiLayerOutputs])

			temp_fiLayerOutputs = fiLayerOutputs[0]
			self.logger.write("Faulty Layer [" + str(randnum) + "]: \n\n")
			self.logger.write(str(temp_fiLayerOutputs) + "\n")
			self.logger.write("\nCompleted injections... exiting" + "\n")
			self.logger.close()
			# Uncomment below line and comment next two lines for ImageNet models
			# return pred
			labels = np.argmax(pred, axis=-1)
			return labels[0]

		elif(fiConf["Mode"] == "multiple"):

			""" Multiple layer fault injection mode """

			self.logger.write(str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))+":" + "\n")
			self.logger.write("----------------Starting fault injection in all layers----------------\n\n")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			self.logger.write("Type of Fault: " + str(fiConf["Type"]) + "\n")
			self.logger.write("Amount of Faults: " + str(fiConf["Amount"]) + "\n")

			# Get the input for which dynamic injection is to be done
			x_test = kwargs["x_test"]

			# Get the outputs of the first layer
			get_output_0 = K.function([model.layers[0].input], [model.layers[1].output])
			fiLayerOutputs = get_output_0([x_test])

			self.logger.write("\n Original Layer [" + str(randnum) + "] Outputs: \n\n")
			self.logger.write(str(get_output([x_test])) + "\n")

			# Loop through each available layer in the model
			for n in range(1, len(model.layers) - 2):

				# Unstack elements into a single dimension
				elem_shape = fiLayerOutputs[0].shape
				fiLayerOutputs[0] = fiLayerOutputs[0].flatten()
				num = fiLayerOutputs[0].shape[0]
				if(fiFault == "zeros"):
					fiSz = (fiSz * num) / 100
					fiSz = math.floor(fiSz)

				# Choose the indices for FI
				#ind = random.sample(range(num), fiSz)
				ind = np.random.choice(range(num), fiSz, replace=True)
				self.logger.write("\nLayer Output Indicies (#s) to Inject: " + str(ind) + "\n\n")

				# Inject the specified fault into the randomly chosen values
				item_counter = 0

				if(fiFault == "zeros"):
					for item in ind:
						self.logger.write("("+ str(item_counter) +") Original Output #" + str(item) + " Value: " + str(float(fiLayerOutputs[0][item])) + "\n")
						fiLayerOutputs[0][item] = 0.
						item_counter+=1
				elif(fiFault == "random"):
					for item in ind:
						self.logger.write("("+ str(item_counter) +") Original Output #" + str(item) + " Value: " + str(float(fiLayerOutputs[0][item])) + "\n")
						fiLayerOutputs[0][item] = np.random.random()
						self.logger.write("("+ str(item_counter) +") Faulty Output #" + str(item) + " Value:   " + str(fiLayerOutputs[0][item]) + "\n")
						item_counter+=1
				elif(fiFault == "bitflips"):
					for item in ind:
						val = fiLayerOutputs[0][item]
						if(fiConf["Bit"] == "N"):
							pos = random.randint(0, 31)
							self.logger.write("("+ str(item_counter) +") Bit Position in 32-Bit Output #" + str(item) + " to be Flipped: " + str(pos) + "\n")
						else:
							pos = int(fiConf["Bit"])
							self.logger.write("("+ str(item_counter) +") Bit Position in 32-Bit Output #" + str(item) + " to be Flipped:" + str(pos) + "\n")

						val_ = bitflip(val, pos)
						fiLayerOutputs[0][item] = val_

						self.logger.write("("+ str(item_counter) +") Original #" + str(item) + " Value: " + str(float(val)) + "\n")
						self.logger.write("("+ str(item_counter) +") Faulty #" + str(item) + " Value:   " + str(val_) + "\n")
						self.logger.write("\n")
						item_counter+=1

				# Reshape into original dimensions
				fiLayerOutputs[0] = fiLayerOutputs[0].reshape(elem_shape)

				"""
				Check if last but one layer reached;
				if not, replace fiLayerOutputs with the next prediction to continue
				"""
				if(n != (len(model.layers) - 3)):
					get_output = K.function([model.layers[n+1].input], [model.layers[n+2].output])
					fiLayerOutputs = get_output([fiLayerOutputs])

				# Get final prediction
				get_pred = K.function([model.layers[len(model.layers)-1].input], [model.layers[-1].output])
				pred = get_pred([fiLayerOutputs])

				temp_fiLayerOutputs = fiLayerOutputs[0]
				self.logger.write("Faulty Layer [" + str(randnum) + "]: \n\n")
				self.logger.write(str(temp_fiLayerOutputs) + "\n")
				self.logger.write("\nCompleted injections... exiting")
				self.logger.close()
				# Uncomment below line and comment next two lines for ImageNet models
				# return pred
				labels = np.argmax(pred, axis=-1)
				return labels[0]
