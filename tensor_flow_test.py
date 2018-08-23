import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import holdem_calc

#different ideas
#1 - use hand outcome as label
#2 - use round outcome as label (4 models)
#opt for 2 first

frames = []

#read in data
for sheet in listdir('/Users/johnled/Documents/vbox_share_folder/trend-ai-mikethemouth/game_data'):
    df = pd.read_csv(('/Users/johnled/Documents/vbox_share_folder/trend-ai-mikethemouth/game_data/' + sheet), usecols=range(0,9), header=None)
    frames.append(df)    

sampleData = pd.concat(frames)

#basic definitions
card_to_score = {
    "2" : 2,
    "3" : 3,
    "4" : 4,
    "5" : 5,
    "6" : 6,
    "7" : 7,
    "8" : 8,
    "9" : 9,
    "T" : 10,
    "J" : 11,
    "Q" : 12,
    "K" : 13,
    "A" : 14
}

actions = {
    "fold" : 0,
    "check" : 1,
    "call" : 2, 
    "bet" : 3,
    "raise" : 4, 
    "allin" : 5
}

def repl(tableHand):
    tableHandChars = list(tableHand)
    for i in range(0, len(tableHandChars)):
        c = tableHandChars[i]
        if c in card_to_score:
            replC = card_to_score[c]
            tableHand = tableHand.replace(c, str(replC))
    return tableHand

def replAction(playerHand):
    if playerHand in actions:
        return actions[playerHand]


def encodeSuit(tableHand):
    tableHandChars = list(tableHand)
    for i in range(0, len(tableHandChars)):
        c = tableHandChars[i]
        if c in ["d", "c", "s", "h"]:
            replC = ord(c)
            tableHand = tableHand.replace(c, str(replC))
    return tableHand
"""
def encodeAction(action):
    actionChars = list(action)
    for i in range(0, len(actionChars)):
        c = actionChars[i]
        replC = ord(c)
        action = action.replace(c, str(replC))
    return action
"""

def formatHand(hand):
    #compute the most probable hand for each player, see if pattern emerges between hand an action in deal round
    #note that cards must be given in the following format: As, Jc, Td, 3h
    hand = hand.replace(".", ",")
    #print hand
    handChars = list(hand)
    for i in range(0, len(handChars)):
        c = handChars[i]
        if c in ["D", "C", "S", "H"]:
            replC = c.lower()
            hand = hand.replace(c, str(replC))
    card1, card2 = hand.split(",")
    hand = [card1, card2]
    return hand

#adapt to fill in ["?", "?"], based on number of players in game
def createLikelyHandSeries(rowPosition, TrainingSetHoleCards, TrainingSetTableCards=None):
    #take in hand, compute likely outcome for each hand
    if TrainingSetTableCards is None:
        #handStats = holdem_calc.calculate(['Ah', 'Td', '5c'], True, 1, None, ['9d', ' 7d'], True)
        #winningPercentages, playerHistograms = holdem_calc.calculate(None, True, 1, None, TrainingSetHoleCards, True)
        #print winningPercentages, playerHistograms
        print holdem_calc.calculate(None, True, 1, None, TrainingSetHoleCards, True)
    return None
#class player (self, sampleData):

#discount sequence of actions within round for now
dealSampleSet = sampleData.loc[sampleData[1] == "Deal"]
dealSampleSet = dealSampleSet.loc[:, [3, 4, 5, 6, 7]]
dealSampleTrainSet = dealSampleSet.iloc[range(0, 999), :]
dealSampleTrainSet.iloc[:, 0] = dealSampleTrainSet.iloc[:, 0].apply(lambda x: replAction(x) if pd.notna(x) else x)
dealSampleTrainSet.iloc[:, 3] = dealSampleTrainSet.iloc[:, 3].apply(lambda x: formatHand(x) if pd.notna(x) else x)
createLikelyHandSeries(1, dealSampleTrainSet.iloc[1,3])
#dealSampleTrainSet.iloc[:, 3] = dealSampleTrainSet.iloc[1, 3].apply(lambda x: createLikelyHandSeries(x) if pd.notna(x) else x)
print ('size of deal dataset is {}'.format(len(dealSampleTrainSet)))
print dealSampleTrainSet
#at this point it is suggested to normalize data, skipping for now...





#attributes = action -- encode as hex, cost, chipCount, tableCards(list of pairs) -- encode as hex
#label = holeCards -- as a pair(integer, character) -- encode as hex
#hands (increasing order): [highCard, pair, twoPair, threeOfaKind, straight, flush, fullHouse, fourOfaKind, straightFlush]

#formattedHand = (dealSampleSet.loc[dealSampleSet[6]].lower()).replace(".","")
#print formattedHand
#print ord(formattedHand)

"""
/Users/johnled/Documents/vbox_share_folder/trend-ai-mikethemouth/lib/python2.7/site-packages/pandas/core/indexing.py:543: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  self.obj[item] = s
"""


#print attributes
#attributes.iloc[:, 3] = attributes.iloc[:, 3].str.replace(".","")
#attributes.iloc[:, 3] = attributes.iloc[:, 3].apply(lambda x: repl(x) if pd.notna(x) else x)
#attributes.iloc[:, 3] = attributes.iloc[:, 3].apply(lambda x: encodeSuit(x) if pd.notna(x) else x)
#attributes.iloc[:, 0] = attributes.iloc[:, 0].apply(lambda x: encodeAction(x) if pd.notna(x) else x)
#print attributes

"""
here we are trying to predict the hand value -- according to the hold 'em calculator, 
based on action + cost + round
use regression from tensorflow, but need 
"""