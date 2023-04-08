'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-04-04 21:45:26
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-04-08 22:48:08
FilePath: \知识图谱作业2\BiLSTM-CRF.py
Description: coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn

Copyright (c) 2023 by zhangynag0207@bupt.edu.cn, All Rights Reserved. 
'''
import json

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def argmax(vec):
	# 得到最大的值的索引
	_, idx = torch.max(vec, 1)
	return idx.item()

def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]  # max_score的维度为1
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # 维度为1*5
	return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
	#等同于torch.log(torch.sum(torch.exp(vec)))，防止e的指数导致计算机上溢

class BiLSTM_CRF(nn.Module):
	def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
		super(BiLSTM_CRF, self).__init__() # 继承父类的初始化方法
		self.embedding_dim = embedding_dim # 词嵌入维度
		self.hidden_dim = hidden_dim # LSTM的隐层维度
		self.vocab_size = vocab_size # 词典大小
		self.tag_to_ix = tag_to_ix # {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-PER': 3, 'I-PER': 4}
		self.tagset_size = len(tag_to_ix) # 5

		self.word_embeds = nn.Embedding(vocab_size, embedding_dim) # 词嵌入层
		self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True) # BiLSTM
		self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size) # 将BiLSTM的输出转换为发射矩阵的维度
		# 转移矩阵，transitions[i][j]表示从label_j转移到label_i的概率,虽然是随机生成的但是后面会迭代更新
		self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

		self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # 从任何标签转移到START_TAG不可能
		self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000  # 从STOP_TAG转移到任何标签不可能

		self.hidden = self.init_hidden() # 随机初始化LSTM的输入(h_0, c_0)

	def init_hidden(self):
		return (torch.randn(2, 1, self.hidden_dim // 2), # h_0
				torch.randn(2, 1, self.hidden_dim // 2)) # c_0

	def _forward_alg(self, feats):
		'''
		输入：发射矩阵(emission score)，实际上就是LSTM的输出——sentence的每个word经BiLSTM后，对应于每个label的得分
		输出：所有可能路径得分之和/归一化因子/配分函数/Z(x)
		'''
		init_alphas = torch.full((1, self.tagset_size), -10000.) # 1*5
		init_alphas[0][self.tag_to_ix[START_TAG]] = 0. # 1*5s

		# 包装到一个变量里面以便自动反向传播
		forward_var = init_alphas
		for feat in feats: # w_i
			alphas_t = []
			for next_tag in range(self.tagset_size): # tag_j
				# t时刻tag_i emission score（1个）的广播。需要将其与t-1时刻的5个previous_tags转移到该tag_i的transition scors相加
				emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size) # 1*5
				# t-1时刻的5个previous_tags到该tag_i的transition scors
				trans_score = self.transitions[next_tag].view(1, -1)  # 维度是1*5
				next_tag_var = forward_var + trans_score + emit_score
				# 求和，实现w_(t-1)到w_t的推进
				alphas_t.append(log_sum_exp(next_tag_var).view(1))
			forward_var = torch.cat(alphas_t).view(1, -1) # 1*5

		# 最后将最后一个单词的forward var与转移 stop tag的概率相加
		terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
		alpha = log_sum_exp(terminal_var)
		return alpha

	def _get_lstm_features(self, sentence):
		'''
		输入：id化的自然语言序列
		输出：序列中每个字符的Emission Score
		'''
		self.hidden = self.init_hidden() # (h_0, c_0)
		embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
		lstm_out, self.hidden = self.lstm(embeds, self.hidden)
		lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
		lstm_feats = self.hidden2tag(lstm_out) # len(s)*5
		return lstm_feats

	def _score_sentence(self, feats, tags):
		'''
		输入：feats——emission scores；tags——真实序列标注，以此确定转移矩阵中选择哪条路径
		输出：真实路径得分
		'''
		score = torch.zeros(1)
		# 将START_TAG的标签３拼接到tag序列最前面
		tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
		for i, feat in enumerate(feats):
			score = score + \
					self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
		score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
		return score

	def _viterbi_decode(self, feats):
		# 预测序列的得分，维特比解码，输出得分与路径值
		backpointers = []

		init_vvars = torch.full((1, self.tagset_size), -10000.)
		init_vvars[0][self.tag_to_ix[START_TAG]] = 0

		forward_var = init_vvars
		for feat in feats:
			bptrs_t = []
			viterbivars_t = []

			for next_tag in range(self.tagset_size):
				next_tag_var = forward_var + self.transitions[next_tag]  # forward_var保存的是之前的最优路径的值
				best_tag_id = argmax(next_tag_var)  # 返回最大值对应的那个tag
				bptrs_t.append(best_tag_id)
				viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

			forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
			backpointers.append(bptrs_t)  # bptrs_t有５个元素

		# 其他标签到STOP_TAG的转移概率
		terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
		best_tag_id = argmax(terminal_var)
		path_score = terminal_var[0][best_tag_id]

		best_path = [best_tag_id]
		for bptrs_t in reversed(backpointers):
			best_tag_id = bptrs_t[best_tag_id]
			best_path.append(best_tag_id)
		
		# 无需返回最开始的START位
		start = best_path.pop()
		assert start == self.tag_to_ix[START_TAG]
		best_path.reverse()  # 把从后向前的路径正过来
		return path_score, best_path

	def neg_log_likelihood(self, sentence, tags):  # 损失函数
		feats = self._get_lstm_features(sentence)  # len(s)*5
		forward_score = self._forward_alg(feats)  # 规范化因子/配分函数
		gold_score = self._score_sentence(feats, tags) # 正确路径得分
		return forward_score - gold_score  # Loss（已取反）

	def forward(self, sentence):
		'''
		解码过程，维特比解码选择最大概率的标注路径
		'''
		lstm_feats = self._get_lstm_features(sentence)

		score, tag_seq = self._viterbi_decode(lstm_feats)
		return score, tag_seq

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 6 
# 由于标签一共有O， B-LOCATION, I-LOCATION， B-TIME，START_TAG ，STOP_TAG6个，所以embedding_dim为6
HIDDEN_DIM = 4 # 这其实是BiLSTM的隐藏层的特征数量，因为是双向所以是2倍，单向为2

data = json.load(open('./data/data.json', 'r', encoding='utf-8'))
print(data[1])

# training_data = [(
# 	"the wall street journal reported today that apple corporation made money".split(),
# 	"B I I I O O O B I O O".split()
# ), (
# 	"georgia tech is a university in georgia".split(),
# 	"B I O O O O B".split()
# )]

word_to_ix = {}
for piece in data:
	for word in piece["text"]:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"O": 0, "B-LOCATION": 1, "I-LOCATION": 2, "B-TIME": 3, "I-TIME": 4, START_TAG: 5, STOP_TAG: 6}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
print(model)

i = 0
for epoch in range(1):
	for piece in data: # 对每个句子进行训练
		sentence = piece["text"] # 句子
		tags = piece["label"] # 标签
		model.zero_grad() # 梯度清零
		
        # 输入
		sentence_in = prepare_sequence(sentence, word_to_ix) # 将句子转化为索引
		targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long) # 将标签转化为索引
		
        # 获取loss
		loss = model.neg_log_likelihood(sentence_in, targets) # 计算损失
		print(i/len(data), "\tloss:", loss.item()) # 打印损失
		i += 1 
        # 反向传播
		loss.backward() # 反向传播
		optimizer.step() # 更新参数

model.eval()
# 保存模型
torch.save(model, './model.pkl')

with torch.no_grad():
	precheck_sent = prepare_sequence(data[0]["text"], word_to_ix)
	print(model(precheck_sent))
