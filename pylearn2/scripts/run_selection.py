#!/usr/bin/env python

import os
import argparse
import numpy

from pylearn2.testing import skip
from pylearn2.testing import no_debug_mode
from pylearn2.config import yaml_parse
from pylearn2.utils import serial

import gc
import sys
import time
import re
import copy
import shutil
import glob
import copy

def is_float(s):
	try:
		float(s)
		if len(s.split('.')) == 2:
			return True
		else:
			return False
	except ValueError:
		return False

def is_int(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

def convert_seconds_to_str(seconds):
	time_string = ""
	if seconds < 60:
		time_string = str(seconds) + " sec."
	elif seconds >= 60 and seconds < 3600:
		minutes = int(seconds / 60)
		seconds = int((seconds / 60 - minutes) * 60)
		time_string = str(minutes) + " min. " + str(seconds) + " sec."
	else:
		hours = int(seconds / 3600)
		minutes = int((seconds / 3600 - hours) * 60)
		seconds = int(((seconds / 3600 - hours) * 60 - minutes) * 60)
		time_string = str(hours) + " hr. " + str(minutes) + " min. " + str(seconds) + " sec."

	return time_string

class SummaryGenerator(object):
	def __init__(self, file_name, report_cf_pairs, sort_by_pair = 0):
		self.file_name = file_name
		self.report_cf_pairs = report_cf_pairs
		self.sort_by_pair = sort_by_pair

		self.stored_results = []
		self.timings = []

	def append(self, file_directory, param_summary_name):
		results = []
		results.append(len(self.stored_results) + 1)
		results.append(param_summary_name)
		for model_file, channel_name in self.report_cf_pairs:
			model = serial.load(file_directory + model_file)
			monitor = model.monitor
			del model
			gc.collect()
			channels = monitor.channels

			error = float(channels[channel_name].val_record[-1])
			results.append(error)

		self.stored_results.append(results)

		error_file = open(self.file_name, 'w')
		sorted_results = sorted(self.stored_results, key=lambda x: x[self.sort_by_pair + 2])

		out_string = "Num."
		out_string = out_string + " " * (10 - len(out_string))
		error_file.write(out_string)
		for model_file, channel_name in self.report_cf_pairs[1:]:
			out_string = channel_name
			out_string = out_string + " " * (20 - len(out_string))
			error_file.write(out_string)

		error_file.write("Parameters\n")

		for result_list in sorted_results:
			out_string = str(result_list[0])
			out_string = out_string + " " * (10 - len(out_string))
			error_file.write(out_string)
			for result in result_list[3:]:
				out_string = str(result)
				out_string = out_string + " " * (20 - len(out_string))

				error_file.write(out_string)

			error_file.write(result_list[1] + "\n")
		error_file.close()

class TrainInstance(object):
	def __init__(self, yaml_file_name, param_set, children = [], parent = None):
		self.yaml_file_name = yaml_file_name
		self.param_set = param_set
		self.children = children
		for child in self.children:
			child.parent = self

		self.parent = parent

		self.name_string = self.yaml_file_name + ": "
		for param_name in self.param_set:
			if param_name != "save_path":
				self.name_string = self.name_string + param_name + " - " + str(self.param_set[param_name]) + " "
		self.name_string = self.name_string[:-1]

	def run(self, summary_generator, total_trainings, verbose = False):
		yaml_code = open(self.yaml_file_name, 'r').read()
		mod_param_set = copy.deepcopy(self.param_set)
		mod_param_set["save_path"] = mod_param_set["save_path"] + "tmp"

		print str(len(summary_generator.timings) + 1) + "/" + str(total_trainings), "training", self.yaml_file_name
		print {key : mod_param_set[key] for key in mod_param_set if key != "save_path"}
		#print self.get_full_name()
		stdout, stderr = sys.stdout, sys.stderr
		if not verbose:
			devnull = open(os.devnull, 'w')
			sys.stdout, sys.stderr = devnull, devnull
		else:
			print ""

		try:
			train = yaml_parse.load(yaml_code % mod_param_set)
			start_time = time.clock()
			train.main_loop()
			stop_time = time.clock()
			summary_generator.timings.append(stop_time - start_time)
		except:
			sys.stdout, sys.stderr = stdout, stderr
			raise
			
		if not verbose:
			sys.stdout, sys.stderr = stdout, stderr
		else:
			print ""

		print "Time for this model:", convert_seconds_to_str(summary_generator.timings[-1]), "Time remaining:", convert_seconds_to_str(numpy.mean(summary_generator.timings) * (total_trainings - len(summary_generator.timings)))

		if len(self.children) > 0:
			print ""
			for child in self.children:
				child.run(summary_generator, total_trainings)
		else:
			full_name = self.get_full_name()

			save_dir = self.param_set["save_path"]
			'''
			numbered_dirs = []
			for directory in [x[0] for x in os.walk(self.param_set["save_path"])]:
				if is_int(directory.split('/')[-1]):
					numbered_dirs.append(int(directory.split('/')[-1]))

			save_num = 1
			if len(numbered_dirs) > 0:
				save_num = max(numbered_dirs) + 1
			'''
			save_num = len(summary_generator.stored_results) + 1
			save_dir = save_dir + str(save_num)
			if not os.path.isdir(save_dir):
				os.mkdir(save_dir)
			else:
				for fl in glob.glob(save_dir + "/*"):
					os.remove(fl)

			for file_name in glob.glob(self.param_set["save_path"] + "tmp/*"):                                                                                                                                 
				shutil.copy(file_name, save_dir + "/")

			summary_generator.append(save_dir + "/", full_name)

			print ""

	def get_full_name(self):
		if self.parent is not None:
			return self.parent.get_full_name() + " | " + self.name_string
		else:
			return self.name_string

class Train(object):
	def __init__(self, params, train_instances, summary_generator):
		self.params = params
		self.train_instances = train_instances
		self.summary_generator = summary_generator
		self.instance_count = self.count_train_instances(self.train_instances)

	def run(self, verbose = False):
		if not os.path.isdir(self.params["GLOBAL"]["save_path"][0] + "tmp"):
			os.makedirs(self.params["GLOBAL"]["save_path"][0] + "tmp")

		for train_instance in self.train_instances:
			for fl in glob.glob(self.params["GLOBAL"]["save_path"][0] + "tmp/*"):
				os.remove(fl)

			train_instance.run(self.summary_generator, self.instance_count, verbose = verbose)

	def count_train_instances(self, train_instance_layer):
		instance_count = len(train_instance_layer)
		for train_instance in train_instance_layer:
			instance_count += self.count_train_instances(train_instance.children)

		return instance_count

class ParameterParser(object):
	def __init__(self, file_name):
		self.params, tied_training_files = self.load_parameters(file_name)

		if self.params["GLOBAL"]["save_path"][0][-1] != "/":
			self.params["GLOBAL"]["save_path"][0] = self.params["GLOBAL"]["save_path"][0] + "/"

		file_sequence = []
		if "training_order" in self.params["META_PARAMS"]:
			file_sequence = self.params["META_PARAMS"]["training_order"]

		unordered_files = []
		if "unordered_files" in self.params["META_PARAMS"]:
			unordered_files = self.params["META_PARAMS"]["unordered_files"]
			
		params_by_file = {}
		for yaml_file_name in file_sequence + unordered_files:
			joined_params = None
			if "GLOBAL" in self.params and yaml_file_name in self.params:
				joined_params = dict(self.params["GLOBAL"].items() + self.params[yaml_file_name].items())
			elif yaml_file_name in self.params:
				joined_params = self.params[yaml_file_name].copy()
			elif "GLOBAL" in self.params:
				joined_params = self.params["GLOBAL"].copy()
			else:
				raise Exception("No parameters for file " + yaml_file_name + " specified in META_PARAMS list :(")
			
			yaml = open(yaml_file_name, 'r').read()
			insertable_params = self.generate_insertable_param_dict(joined_params, yaml)
			params_by_file[yaml_file_name] = self.generate_all_permutations(insertable_params)

		self.train_instances = []
		if len(file_sequence) > 0:
			file_sequence.reverse()
			previous_layer_instances = []
			current_layer_instances = []
			for i, yaml_file_name in enumerate(file_sequence):
				ties = None
				for tie in tied_training_files:
					if yaml_file_name in tie:
						ties = tie

				if ties is None:
					ties = [yaml_file_name]

				for k, param_set in enumerate(params_by_file[yaml_file_name]):
					if len(ties) == 1 or i == 0:
						current_layer_instances.append(TrainInstance(yaml_file_name, param_set, children = copy.deepcopy(previous_layer_instances)))
					else:
						if file_sequence[i - 1] in ties:
							current_layer_instances.append(TrainInstance(yaml_file_name, param_set, children = copy.deepcopy([previous_layer_instances[k]])))
						else:
							current_layer_instances.append(TrainInstance(yaml_file_name, param_set, children = copy.deepcopy(previous_layer_instances)))

				previous_layer_instances = current_layer_instances
				current_layer_instances = []

			self.train_instances = previous_layer_instances

		for yaml_file_name in unordered_files:
			for param_set in params_by_file[yaml_file_name]:
				self.train_instances.append(TrainInstance(yaml_file_name, param_set))

		global_path = self.params["GLOBAL"]["save_path"][0]

		report_cf_pairs = []
		report_cf_pairs.append((self.params["META_PARAMS"]["sort_by_model"][0], self.params["META_PARAMS"]["sort_by_channel"][0]))
		for report_channel_name in self.params["META_PARAMS"]["report_channels"]:
			report_cf_pairs.append((self.params["META_PARAMS"]["sort_by_model"][0], report_channel_name))

		self.summary_generator = SummaryGenerator(global_path + self.params["META_PARAMS"]["report_output_file"][0], report_cf_pairs)

	def get_train_object(self):
		return Train(self.params, self.train_instances, self.summary_generator)

	def generate_insertable_param_dict(self, param_dict, yaml_code):
		insertable_params = {}

		insert_indices = [substring.start() for substring in re.finditer('%', yaml_code)]
		for insert_index in insert_indices:
			if yaml_code[insert_index + 1] == '(':
				insert_name = None
				start_pos = insert_index + 2
				end_pos = start_pos + 1
				while yaml_code[end_pos] != ')':
					end_pos += 1

				insert_name = yaml_code[start_pos:end_pos]
				if insert_name != "W_lr_scale" and insert_name != "keep_input_constant":
					if insert_name not in param_dict:
						raise Exception("Found a value template in one of your yaml files named \"" + insert_name + "\" with no definition in your conf file.")

					insertable_params[insert_name] = param_dict[insert_name]

		#insertable_params["W_lr_scale"] = insertable_params["h0_drop_prob"][0] * insertable_params["h1_drop_prob"][0]
		return insertable_params


	def generate_all_permutations(self, params):
		permutations = []

		param_it_tracker = {}
		params_exhausted = False
		for key in params:
			param_it_tracker[key] = 0

		increment_next = False
		while not increment_next:
			new_param_set = {}
			for key in params:
				new_param_set[key] = params[key][param_it_tracker[key]]

			increment_next = True
			for key in param_it_tracker:
				if increment_next:
					param_it_tracker[key] += 1
					increment_next = False

				if len(params[key]) <= param_it_tracker[key]:
					param_it_tracker[key] = 0
					increment_next = True

			if "h0_drop_prob" in new_param_set and "h1_drop_prob" in new_param_set:
				new_param_set["W_lr_scale"] = new_param_set["h0_drop_prob"] * new_param_set["h1_drop_prob"]
			if "input_include_prob" in new_param_set:
				if new_param_set["input_include_prob"] > 0.7:
					new_param_set["keep_input_constant"] = 1
				else:
					new_param_set["keep_input_constant"] = 0
			permutations.append(new_param_set)

		return permutations

	def parse_number(self, number_string):
		if is_float(number_string):
			return float(number_string)
		else:
			return int(number_string)

	def parse_value(self, value):
		value_sequence = []
		if len(value.split('->')) == 3:
			start_number, increment, end_number = value.split('->')
			start_number = self.parse_number(start_number.strip())
			end_number = self.parse_number(end_number.strip())

			increment = increment.strip()
			mode = increment[0]
			increment = self.parse_number(increment[1:])

			curr_number = start_number
			value_sequence.append(curr_number)
			inc_it = 0
			while curr_number < end_number:
				inc_it += 1
				if mode == '+':
					curr_number = start_number + increment * inc_it
				elif mode == '*':
					curr_number = start_number * (increment ** inc_it)

				if curr_number < end_number:
					value_sequence.append(curr_number)
		elif value.isdigit():
			value_sequence.append(int(value))
		elif is_float(value):
			value_sequence.append(float(value))
		else:
			if value[0] == "\"" and value[-1] == "\"":
				value_sequence.append(value[1:-1].strip())
			else:
				value_sequence.append(value.strip())

		return value_sequence

	def parse_value_enumeration(self, enum_string):
		values = []
		if len(enum_string) != 0:
			for value_string in enum_string.split(','):
				values.extend(self.parse_value(value_string.strip()))

		return values

	def load_parameters(self, param_file_name):
		param_lines = open(param_file_name, 'r').readlines()
		parameters = {}
		tied_training_files = []
		current_section_names = None
		for line in param_lines:
			if len(line) > 3:
				clean_line = line.strip()
				if clean_line[0] == '[' and clean_line[-1] == ']':
					current_section_name = clean_line[1:-1]
					current_section_names = current_section_name.split(',')
					tied_training_files.append([])
					for i in range(len(current_section_names)):
						current_section_names[i] = current_section_names[i].strip()
						parameters[current_section_names[i]] = {}
						tied_training_files[-1].append(current_section_names[i])
				else:
					param_name, param_values = clean_line.split(':')
					param_name, param_values = param_name.strip(), param_values.strip()

					for current_section_name in current_section_names:
						parameters[current_section_name][param_name] = self.parse_value_enumeration(param_values)

		return parameters, tied_training_files


parser = argparse.ArgumentParser(description='Parameter selection utility for pylearn2. Trains networks with specified parameter sets.')
parser.add_argument('network_variables', help='Config file with varying parameters specified in netwrok_specification')
parser.add_argument('-v', action='store_true', help='Outputs all channel data while learning')

args = parser.parse_args()

parameter_parser = ParameterParser(args.network_variables)

for param_category in parameter_parser.params:
	print "[" + param_category + "]"
	
	for param in parameter_parser.params[param_category]:
		print "  " + param + ":", parameter_parser.params[param_category][param]
	print ""

print "            ...STARTING TRAINING..."
verbose = False
if args.v == True:
	verbose = True

training = parameter_parser.get_train_object()
training.run(verbose = verbose)
