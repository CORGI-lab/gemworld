from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beams = []
SOS_token = 0 #start of sentence
EOS_token = 1 #end of sentence

#Keep track of unique indices per word
#dictionaries for word->index and index->word
class Voc:
	def __init__(self, name):
		self.name = name
		self.word2index = {} #maps word to which number word it is in the vocab (if hello is second word hello-->2)
		self.word2count = {} #tracks how many times a word appears
		self.index2word = {0: "SOS", 1: "EOS"} #opposite of word2index
		self.n_words = 2  # Count SOS and EOS, number of words
	
	#splits sentence into words and adds each word to dict if it isn't already
	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

#adds words to dictionary
	def addWord(self, word):
		#if word is not in dictionary add it to end
		if word not in self.word2index: 
			self.word2index[word] = self.n_words
			self.word2count[word] = 1 #set count of word to 1
			self.index2word[self.n_words] = word
			self.n_words += 1 #increment number of word by 1

		#if word is in dictionary
		else:
			self.word2count[word] += 1 #increment count of word by 1

#read from training file (1st and 2nd line form a pair....)
def readVocs(corpus, corpus_name):
	print("Reading lines...")
	
	lines = open(corpus).\
		read().split('\n')

	pairs = [l.split('\t') for l in lines]
	inputState = Voc("inputState")
	outputState = Voc("ouputState")
	return inputState, outputState, pairs

MAX_LENGTH = 100

#if both elements of the pair are less than 100 words ret 1
#else return 0
def filterPair(p):
	return len(p[0].split(' ')) < MAX_LENGTH and \
		len(p[1].split(' ')) < MAX_LENGTH

#for each pair in pairs, if filterPair-->1 return the pair as an array
def filterPairs(pairs):
	d = -1
	for i in range(len(pairs)):
		if len(pairs[i]) != 2:
			d=i
	if i != -1:
		pairs.pop(i)
	return [pair for pair in pairs if filterPair(pair)]

#read txt w/ trans and make pairs, filter pairs
#for each pair, add first el of pair to inputlang dict and second el of pair to output lang dict
#print num of words for out/input lang
def prepareData(corpus, corpus_name):
	inputState, outputState, pairs = readVocs(corpus, corpus_name)
	print("Read %s sentence pairs" % len(pairs))
	pairs = filterPairs(pairs)
	print("Trimmed to %s sentence pairs" % len(pairs))
	print("Counting words...")
	for pair in pairs:
		inputState.addSentence(pair[0])
		outputState.addSentence(pair[1])
	print(inputState.name, inputState.n_words)
	print(outputState.name, outputState.n_words)
	return inputState, outputState, pairs


# inputState, outputState, pairs = prepareData("training2gems.txt", "train")
inputState, outputState, pairs = prepareData("/Users/adriennecorwin/Research/Training Files/trainingOnlyFuture.txt", "train")

# inputState, outputState, pairs = prepareData("trainingRiver.txt", "train")

print(random.choice(pairs))

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__() #??
		self.hidden_size = hidden_size
		
		#an embedding module containing input_size # of tensors each of size hidden_size??
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size) #??

#make tensor of dim (1,1,-1) apply multilayer to it??
	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1) 
		output = embedded
		output, hidden = self.gru(output, hidden)
		return output, hidden

#returns a tensor filled with zeros (3D? 1x1xhidden?)
	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(self.dropout_p)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)

		attn_weights = F.softmax(
			self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0),
								 encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.gru(output, hidden)

		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)

#split sentence into words and for each word return index of that word stored in Lang dict
def indexesFromSentence(voc, sentence):
	return [voc.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(voc, sentence):
	indexes = indexesFromSentence(voc, sentence) #indexes=all indexes of words from the sentence
	indexes.append(EOS_token)
	#return tensor(multi-dim matrix containing elements (indexes) of a single type  specified by dtype-->long)
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
	input_tensor = tensorFromSentence(inputState, pair[0]) #input tensor=tensor made from original
	target_tensor = tensorFromSentence(outputState, pair[1]) #target tensor=tensor made from translation
	return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	loss = 0

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device=device)

	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
		# Teacher forcing: Feed the target as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
			loss += criterion(decoder_output, target_tensor[di])
			decoder_input = target_tensor[di]  # Teacher forcing

	else:
		# Without teacher forcing: use its own predictions as the next input
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(
				decoder_input, decoder_hidden, encoder_outputs)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()  # detach from history as input

			loss += criterion(decoder_output, target_tensor[di])
			if decoder_input.item() == EOS_token:
				break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(load_file, n_iters, print_every=1000, plot_every=100, save_every=50000, learning_rate=0.001):
	start = time.time()
	plot_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every
	hidden_size = 256
	encoder = EncoderRNN(inputState.n_words, hidden_size).to(device)
	decoder = AttnDecoderRNN(hidden_size, outputState.n_words, dropout_p=0.1).to(device)
	
	if load_file:
		checkpoint = torch.load(load_file)
		encoder.load_state_dict(checkpoint['en'])
		decoder.load_state_dict(checkpoint['de'])

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

	if load_file:
		encoder_optimizer.load_state_dict(checkpoint['en_opt'])
		decoder_optimizer.load_state_dict(checkpoint['de_opt'])

	training_pairs = [tensorsFromPair(random.choice(pairs))
					  for i in range(n_iters)]
	criterion = nn.NLLLoss()
	start_iteration = 1
	count = 0
	
	if load_file:
		start_iteration = checkpoint['iteration'] + 1
	for iter in range(start_iteration, n_iters + 1):
		training_pair = training_pairs[iter - 1]
		input_tensor = training_pair[0]
		target_tensor = training_pair[1]

		loss = train(input_tensor, target_tensor, encoder,
					 decoder, encoder_optimizer, decoder_optimizer, criterion)
		print_loss_total += loss
		plot_loss_total += loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0
		
		if iter % save_every == 0:
			torch.save({
				'iteration': iter,
				'en': encoder.state_dict(),
				'de': decoder.state_dict(),
				'en_opt': encoder_optimizer.state_dict(),
				'de_opt': decoder_optimizer.state_dict(),
				'loss': loss
			}, "modelOnlyFuture"+str(iter)+".tar")
			
	# showPlot(plot_losses)
#	return encoder, decoder

#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
#import matplotlib.ticker as ticker
import numpy as np

#
# def showPlot(points):
# 	plt.figure()
# 	fig, ax = plt.subplots()
# 	# this locator puts ticks at regular intervals
# 	loc = ticker.MultipleLocator(base=0.2)
# 	ax.yaxis.set_major_locator(loc)
# 	plt.plot(points)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
	beams = []
	with torch.no_grad():
		input_tensor = tensorFromSentence(inputState, sentence)
		input_length = input_tensor.size()[0]
		encoder_hidden = encoder.initHidden()

		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

		for ei in range(input_length):
			encoder_output, encoder_hidden = encoder(input_tensor[ei],
													 encoder_hidden)
			encoder_outputs[ei] += encoder_output[0, 0]

		decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

		decoder_hidden = encoder_hidden

		decoded_words = []
		decoder_attentions = torch.zeros(max_length, max_length)
		#print(decoder_attentions)
		beam_size = 10
		candidates = []
		all_candidates = []
		
		for di in range(max_length):
			if di == 0:
				decoder_output, decoder_hidden, decoder_attention = decoder(
					decoder_input, decoder_hidden, encoder_outputs)
				decoder_attentions[di] = decoder_attention.data
				topv, topi = decoder_output.data.topk(beam_size)
				#print(topv)
				#print(topi)
				for k in range(beam_size):
					candidates.append([topi[0][k].unsqueeze(0), topv[0][k].cpu().data.numpy(), decoder_hidden, topv[0][k].cpu().data.numpy()])
				#print(candidates)
			else:
				all_candidates = []
				break_flag = True
				for candidate in candidates:
					if candidate[0][len(candidate[0])-1] != EOS_token:
						break_flag = False
				if break_flag == True:
					break
				for k in range(beam_size):
					candidate = candidates[k]
					if candidate[0][len(candidate[0])-1] == EOS_token:
						all_candidates.append(candidate)
					else:
						decoder_input = candidate[0][len(candidate[0])-1]
						decoder_output, decoder_hidden, decoder_attention = decoder(
							decoder_input, candidate[2], encoder_outputs)
						topv, topi = decoder_output.data.topk(beam_size)
						
						for j in range(beam_size):
							new_candidate = [torch.cat((candidate[0], topi[0][j].unsqueeze(0)), 0), (candidate[3] + topv[0][j].cpu().data.numpy())/(len(candidate[0]) + 1), decoder_hidden, candidate[3] + topv[0][j].cpu().data.numpy()]
							all_candidates.append(new_candidate)
				ordered = sorted(all_candidates, key = lambda tup:tup[1], reverse = True)
				candidates = ordered[:beam_size]
		beam = []
		for i,candidate in enumerate(candidates):
			sampled_ids = candidate[0]
			# print("Beam # " + str(i))
			beam = []
			for id in sampled_ids:
				if id == EOS_token:
					beam.append('<EOS>')
					break
				else:
					beam.append(outputState.index2word[id.item()])
				output_sentence = ' '.join(beam)
			# print(output_sentence,candidate[1])
			beams.append(output_sentence)#, candidate[1]))
		sampled_ids = candidates[0][0]
		for id in sampled_ids:
				if id == EOS_token:
					decoded_words.append('<EOS>')
				else:
					decoded_words.append(outputState.index2word[id.item()])
			

		return decoded_words, decoder_attentions[:di + 1], beams

def evaluateRandomly(encoder, decoder, n=20):
	for i in range(n):
		pair = random.choice(pairs)
		print('>', pair[0])
		print('=', pair[1])
		output_words, attentions, beams = evaluate(encoder, decoder, pair[0])
		output_sentence = ' '.join(output_words)
		print('<', output_sentence)
		print('')


def main():
	hidden_size = 256
	#encoder, decoder = trainIters(None, 500000, print_every=5000)
	# trainIters(None, 500000, print_every=5000)
	model = None
	encoder = EncoderRNN(inputState.n_words, hidden_size).to(device)
	decoder = AttnDecoderRNN(hidden_size, outputState.n_words, dropout_p=0.1).to(device)
	model = torch.load("modelOnlyFuture500000.tar")
	encoder.load_state_dict(model['en'])
	decoder.load_state_dict(model['de'])
	decoded_words, attentions, beams = evaluate(encoder, decoder, "3 4 0 1 1 1 0 1 1 3 4 4 1 1 1 1 0 1 1 I am getting gem 1 with gusto. -1 -1 -1 -1 -1")
	print(beams)
	# evaluateRandomly(encoder, decoder)

if __name__=="__main__":main()

