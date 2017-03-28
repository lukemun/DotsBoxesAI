#
# Minimax emulator model w/ keras for Boxes & Dots
#
# @author Luke Munro
#
##

import keras, csv
import numpy as np
import Player
from KerasUtils import *
from keras.models import Sequential, model_from_json

class KerasAI(Player):
	def __init__(self):
		Player.__init__(self, "Mali")
		self.helperAI = Minmax(24, 0, False)
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights('model.h5')
	self.model = loaded_model
	print("\nloaded model...")

	def getMove(self, game_state2):
	reader = csv.reader(open("game_state.csv"), delimiter=",")
	data = reader.__next__()
	game_state = np.transpose(np.array([[int(x)] for x in data]))
	print (game_state.shape)

	prediction = model.predict(game_state, batch_size=1, verbose=1)[0]
	my_list = prediction.tolist()
	print (my_list)
	trunc_list = [np.round(x, 3) for x in my_list]

	moves = orderMoves(prediction)
	clean_state = np.array([[int(x)] for x in data])
	legal_moves = onlyLegal(moves, clean_state)
	next_moves = formatMoves(legal_moves, makeCommands(3))
	next_move = next_moves[0]
	return next_move


# print (trunc_list)
# print (prediction.shape)
# print (prediction)
# print (np.argmax(prediction))
# print (moves)
# print (legal_moves)
# print (next_moves)
