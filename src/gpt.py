import torch
import math
import random
import json
import requests
import numpy as np
from datetime import datetime
from hftokenizer import HFTokenizer


### Model ######################################################################################
class CustomLinear(torch.nn.Module):

	def __init__(self, input_size, output_size):
		super().__init__()
		self.weight = torch.nn.Parameter(0.01*torch.randn((output_size, input_size)))
		self.bias = torch.nn.Parameter(torch.zeros((output_size,)))

	def forward(self, x):
		return x @ self.weight.T + self.bias


class CustomEmbedding(torch.nn.Module):

	def __init__(self, num_embeddings, embedding_dim):
		super().__init__()
		self.weight = torch.nn.Parameter(0.01*torch.randn((num_embeddings, embedding_dim)))

	def forward(self, x):
		return self.weight[x]


class CustomMHA(torch.nn.Module):

	def __init__(self, d_model, n_heads):
		super().__init__()
		self.d_model = d_model
		self.n_heads = n_heads
		self.qkv = torch.nn.Parameter(0.01*torch.randn((3*d_model, d_model)))
		self.wo = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

	def forward(self, x):
		added_batch = False
		if len(x.shape) == 2:
			added_batch = True
			x = x[None,:,:]

		# queries, keys, and values
		B, S, D = x.shape
		QKV = x @ self.qkv.T # B, S, 3D
		Q, K, V = torch.chunk(QKV, 3, -1)

		# split into multiple heads
		dh = D//self.n_heads
		q_heads = torch.reshape(Q, (B, S, self.n_heads, dh))
		k_heads = torch.reshape(K, (B, S, self.n_heads, dh))
		v_heads = torch.reshape(V, (B, S, self.n_heads, dh))

		# reshape into (B*h, S, dh) so we isolate sequences for each head
		q_heads = torch.transpose(q_heads, 1, 2).reshape((B*self.n_heads, S, dh))
		k_heads = torch.transpose(k_heads, 1, 2).reshape((B*self.n_heads, S, dh))
		v_heads = torch.transpose(v_heads, 1, 2).reshape((B*self.n_heads, S, dh))

		# make attention mask
		mask = torch.ones((S,S))
		mask = torch.tril(mask)
		mask = mask[None, :, :]
		mask = mask.to(x.device)

		# attention
		k_heads_t = torch.transpose(k_heads, 1, 2)
		qkt = torch.matmul(q_heads, k_heads_t) / math.sqrt(float(dh))
		qkt = qkt*mask
		qkt[qkt==0] = float('-inf')
		attn = torch.nn.functional.softmax(qkt, dim=-1)
		x = torch.matmul(attn, v_heads)

		# shmush back into the correct shape
		x = torch.reshape(x, (B, self.n_heads, S, dh))
		x = torch.transpose(x, 1, 2) # B, S, h, dh
		x = torch.reshape(x, (B, S, D))

		# apply projection
		x = x @ self.wo.T

		if added_batch:
			x = x[0]

		return x


class TransformerDecoderBlock(torch.nn.Module):

	def __init__(self, d_model, n_heads):
		super().__init__()
		self.norm1 = torch.nn.LayerNorm((d_model,))
		self.mha = CustomMHA(d_model, n_heads)
		self.norm2 = torch.nn.LayerNorm((d_model,))
		self.fc1 = CustomLinear(d_model, 4*d_model)
		self.act = torch.nn.ReLU()
		self.fc2 = CustomLinear(4*d_model, d_model)
		self.dropout = torch.nn.Dropout(0.1)

	def forward(self, x):
		x = x + self.mha(self.norm1(x))
		x = x + self.dropout(self.fc2(self.act(self.fc1(self.norm2(x)))))
		return x
		

class GPTModel(torch.nn.Module):

	def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len):
		super().__init__()

		self.word_embeddings = CustomEmbedding(vocab_size, d_model)
		self.position_embeddings = CustomEmbedding(max_seq_len, d_model)

		self.layers = torch.nn.ModuleList()
		for i in range(layers):
			block = TransformerDecoderBlock(d_model, n_heads)
			self.layers.append(block)

		self.fc_out = CustomLinear(d_model, vocab_size)

	def forward(self, x):
		B, S = x.shape
		positions = torch.arange(S).to(torch.long).to(x.device)
		positions = positions[None, :]
		positions = positions.repeat(B, 1)

		w_emb = self.word_embeddings(x)
		p_emb = self.position_embeddings(positions)
		x = w_emb + p_emb

		for layer in self.layers:
			x = layer(x)

		logits = self.fc_out(x)

		return logits


### Setup ###############################################################################

SCRATCH_GPT_NAME = "scratch-gpt"
SCRATCH_GPT_ENDPOINTS = {
    "http://localhost:11434/api/generate",
    "http://127.0.0.1:11434/api/generate",
}

_real_requests_post = requests.post
scratch_backend = None

def configure_scratch_backend(backend):
    global scratch_backend
    scratch_backend = backend

class Sampler:
    def __init__(self, top_p=0.9):
        self.top_p = float(top_p)

    def sample(self, logits, temperature=1.0):
        logits = logits.astype(np.float64)

        if temperature is None:
            temperature = 1.0
        temperature = float(max(temperature, 1e-6))

        if temperature <= 1e-6:
            return int(np.argmax(logits))

        logits = logits / temperature
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)

        # nucleus sampling
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumulative = np.cumsum(sorted_probs)
        keep = cumulative <= self.top_p
        if not np.any(keep):
            keep[0] = True
        else:
            first_over = np.argmax(cumulative > self.top_p)
            keep[first_over] = True

        kept_idx = sorted_idx[keep]
        kept_probs = sorted_probs[keep]
        kept_probs = kept_probs / kept_probs.sum()

        return int(np.random.choice(kept_idx, p=kept_probs))

class ScratchGPTBackend:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GPTModel(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            layers=config['layers'],
            vocab_size=config['vocab_size'],
            max_seq_len=config['max_seq_len'],
        )
        state = torch.load(config['weights_path'], map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = HFTokenizer()
        tok_cls = type(self.tokenizer.tokenizer)
        self.tokenizer.tokenizer = tok_cls.from_pretrained(config['tokenizer_dir'])
        self.eos_token_id = self.tokenizer.tokenizer.eos_token_id
        self.max_seq_len = int(config['max_seq_len'])
        self.sampler = Sampler(top_p=0.9)

    def generate_text(self, prompt, num_predict=96, temperature=0.7):
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) >= self.max_seq_len:
            token_ids = token_ids[-(self.max_seq_len - 1):]

        generated = []
        seq = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for _ in range(int(num_predict)):
                logits = self.model(seq)[0, -1, :].detach().cpu().numpy()
                next_tok = self.sampler.sample(logits, temperature=temperature)
                if next_tok == self.eos_token_id:
                    break
                generated.append(next_tok)

                next_tok_t = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
                seq = torch.cat([seq, next_tok_t], dim=1)
                if seq.shape[1] >= self.max_seq_len:
                    seq = seq[:, -(self.max_seq_len - 1):]

        return self.tokenizer.decode(generated).strip()

class FakeOllamaResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.ok = True
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=False):
        line = json.dumps(self._payload)
        if decode_unicode:
            yield line
        else:
            yield line.encode('utf-8')

def _extract_url(url):
    if isinstance(url, str):
        return url
    return getattr(url, 'url', str(url))

def fake_post(url, *args, **kwargs):
    url_str = _extract_url(url)

    if url_str in SCRATCH_GPT_ENDPOINTS:
        body = kwargs.get('json')
        if body is None and 'data' in kwargs and kwargs['data'] is not None:
            data = kwargs['data']
            body = json.loads(data if isinstance(data, str) else data.decode('utf-8'))
        if body is None:
            body = {}

        model_name = body.get('model', '')
        if model_name != SCRATCH_GPT_NAME:
            # Any non-scratch model still goes to the real handler.
            return _real_requests_post(url, *args, **kwargs)

        options = body.get('options') or {}
        prompt = body.get('prompt', '')
        num_predict = body.get('num_predict', options.get('num_predict', 96))
        temperature = body.get('temperature', options.get('temperature', 0.7))

        generated = scratch_backend.generate_text(
            prompt=prompt,
            num_predict=num_predict,
            temperature=temperature,
        )

        payload = {
            'model': SCRATCH_GPT_NAME,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'response': generated,
            'done': True,
        }
        return FakeOllamaResponse(payload)

    return _real_requests_post(url, *args, **kwargs)

if __name__ == "__main__":

	model = GPTModel(128, 8, 4, 1000, 512)
	B = 32
	S = 48
	x = torch.randint(1000, (B, S))
	y = model(x)
