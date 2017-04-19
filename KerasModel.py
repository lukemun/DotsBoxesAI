#
# Minimax emulator model w/ keras for Boxes & Dots
#
# @author Luke Munro
#
##

from Player import Player
from Minimax import Minimax	

import keras, csv
import numpy as np
from KerasUtils import *
from keras.models import Sequential, model_from_json


class KerasAI(Player):
	def __init__(self, grid_size):
		Player.__init__(self, "Shallow Blue AI")
		self.grid_size = grid_size
		self.helperAI = Minimax(grid_size, 0, False)
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights('model.h5')
		self.model = loaded_model
		print("\nloaded model...")

	def getMove(self, game_state):
		total_moves = 2*(self.grid_size**2+self.grid_size) 
		made_moves = sum([x for row in game_state for x in row])
		if not self.helperAI.ENDING_SEQUENCE:
			self.helperAI.check_ending_chain(game_state, self.getScore())
		if made_moves < 5:
			next_move = self.helperAI.getMove(game_state, 2)
		elif made_moves < 9 or self.helperAI.ENDING_SEQUENCE:
			next_move = self.helperAI.getMove(game_state, 3)
		else:
			clean_game_state = cleanData(game_state)
			# print (clean_game_state)
			prediction = self.model.predict(np.transpose(clean_game_state), batch_size=1, verbose=1)[0]
			my_list = prediction.tolist()
			# print (my_list)
			trunc_list = [np.round(x, 3) for x in my_list]

			moves = orderMoves(prediction)
			legal_moves = onlyLegal(moves, clean_game_state)
			next_moves = formatMoves(legal_moves, makeCommands(3))
			next_move = next_moves[0]
		return next_move



