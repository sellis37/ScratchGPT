'''
Run this script on its own to train a tokenizer. During training, you can
create an instance of this class and load in the trained tokenization with
the load() method.

For module 6 onward we are using huggingface tokenizers in the interest of speed.
These are heavily optimized to tokenize large amounts of text quickly.

This tokenizer is similar to the tokenizer we trained in module 2, except:
- Uses Ġ to denote space (if you look in the outputs from saving)
- Byte-level BPE
- does a better job of preserving whitespace sequences

'''

from transformers import AutoTokenizer

class HFTokenizer():

	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
		self.tokenizer.eos_token = '<|endoftext|>'

	def train(self, datafile):
		self.tokenizer = self.tokenizer.train_new_from_iterator(
			open(datafile, 'r', encoding='utf-8').readlines(),
			10000,
			limit_alphabet=500,
		)
		self.tokenizer.save_pretrained('./hftokenizer/')

	def load(self):
		self.tokenizer = AutoTokenizer.from_pretrained('./hftokenizer/')

	# string to token_ids
	def encode(self, string):
		return self.tokenizer(string)['input_ids']

	# token_ids to string
	def decode(self, list_of_ids):
		return self.tokenizer.decode(list_of_ids)


if __name__ == '__main__':

	tokenizer = HFTokenizer()
	tokenizer.train('./data.txt')