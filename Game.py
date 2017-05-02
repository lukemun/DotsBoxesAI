# Boxes and Dots Game
#
# Author: Luke Munro
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import KerasModel
import BoxesDots as BD
from Player import Player
from Minimax import Minimax
import utils as UTIL
import sys as SYS

import time, random
import numpy as np
import copy, csv


def main():
	dim = 3 
	num_games = int(input("How many games: "))
	mode = int(input("Player 1 is?\n You (0) | Minimax (1): "))
	if mode == 1:
		base = input("Minimax bonus depth: ")
	mode2 = int(input("Who are you playing?\n Shallow Blue AI (0) | Minimax (1) | Player (2): "))
	if mode2 == 1:
		base2 = int(input("Minimax bonus depth: "))
	
	numMoves = 2*(dim**2+dim)

	if mode == 1:
		player1 = Minimax(dim, base, True)
	elif mode == 0:
		player1 = Player("Human Player1")
	else:
		print ("Unknown command.")
	if mode2 == 1:
		player2 = AI = Minimax(dim, base2, False)
	elif mode2 == 0:
		print ("attempting to load model...")
		try:
			player2 = AI = KerasModel.KerasAI(dim)
			print ("sucess!")
		except:
			print ("failed")
			print ("Make sure you have mode.json & model.h5.")
			raise SystemExit
	elif mode2 == 2:
		player2 = Player("Human Player2")
	else:
		print ("Unknown command.")

	G = BD.BoxesDots(dim, [player1, player2])
	mali_wins = 0
	for i in range(num_games):
		turns = random.randint(0,1) 

		print (G.players[turns].getName() + " starts\n")
		G.display_game()
		# GAME LOOP
		while G.game_status():
			cPlayer = G.players[turns%2]
			check = cPlayer.getScore()
			# For communicating with keras which runs in python3
			# with open("game_state.csv", "wb") as f:
			# 	flate_game_state = [i for row in G.game_state for i in row]
			# 	writer = csv.writer(f)
			# 	writer.writerow(flate_game_state)
			# Back to the good stuff here
			print (cPlayer.getName() + " your move")
			G.turn(cPlayer)
			new_state = copy.deepcopy(G.game_state) # BREAKING CONNECTION
			print (cPlayer.getName() + " move - " + str(cPlayer.last_move))
			G.display_game()
			print (cPlayer.getName() + " your score is " + str(cPlayer.getScore()) + "\n")
			print ("---- Next Move ----")
			if check == cPlayer.getScore():
			 	turns += 1
		if player2.getScore() > player1.getScore():
			mali_wins += 1
		G.show_results()
		G.reset()
	print ("AI wins = " + str(mali_wins))
	print ("Player1 wins = " + str(num_games - mali_wins))
	print ("Done playing.")
	print ("Exiting...")


if __name__ == "__main__":
	main()
