"""
Bin Hoang - University of Rochester

tissue_prior_net.py

Train and save a deep neural network using a dictionary to predict tissue type given b0 and FLAIR prior.
When testing mode is on, use the saved neural network to infer the tissue type.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using CUDA:", torch.cuda.is_available())

# ----------------------------
# DNN Network Definition
# ----------------------------
class Net(nn.Module):
	def __init__(self, input_dim=6, hidden_dim=16, num_classes=7, depth=2, dropout_rate=0.2):
		super(Net, self).__init__()
		fc_layers = []
		for i in range(depth):
			fc_layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
			fc_layers.append(nn.BatchNorm1d(hidden_dim))
			fc_layers.append(nn.ReLU())
			if i < (depth - 1):
				fc_layers.append(nn.Dropout(dropout_rate))
		fc_layers.append(nn.Linear(hidden_dim, num_classes))
		self.fc_layers = nn.Sequential(*fc_layers)

	def forward(self, x):
		return self.fc_layers(x)

# ----------------------------
# DNN Training Function
# ----------------------------
def net_train(
	dictionary_path,
	input_dim=6,
	hidden_dim=16,
	num_classes=7,
	depth=2,
	dropout_rate=0.2,
	lr=1e-3,
	num_epochs=5,
	batch_size=128,
	verbose=False,
	model_save_path=None,
	eval_on_test=False
):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	data = np.load(dictionary_path)

	X_raw = np.stack([
		data['b0'], data['b0_mean'], data['b0_max'],
		data['FLAIR'], data['FLAIR_mean'], data['FLAIR_max']
	], axis=1).astype(np.float32)

	y = data['tissue_label'].astype(np.int64)

	# Remap labels to 0-based
	unique_labels = np.unique(y)
	label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
	y = np.vectorize(label_mapping.get)(y)
	num_classes = len(unique_labels)

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X_raw)

	X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
	X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42)

	X_train = torch.tensor(X_train, dtype=torch.float32)
	X_val = torch.tensor(X_val, dtype=torch.float32)
	X_test = torch.tensor(X_test, dtype=torch.float32)
	y_train = torch.tensor(y_train, dtype=torch.long)
	y_val = torch.tensor(y_val, dtype=torch.long)
	y_test = torch.tensor(y_test, dtype=torch.long)

	train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

	model = Net(input_dim, hidden_dim, num_classes, depth, dropout_rate).to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(num_epochs):
		model.train()
		for X_batch, y_batch in train_loader:
			X_batch = X_batch.to(device)
			y_batch = y_batch.to(device)

			optimizer.zero_grad()
			loss = criterion(model(X_batch), y_batch)
			loss.backward()
			optimizer.step()

	model.eval()
	val_correct = 0
	val_total = 0
	with torch.no_grad():
		for X_batch, y_batch in val_loader:
			X_batch = X_batch.to(device)
			y_batch = y_batch.to(device)
			pred = model(X_batch).argmax(dim=1)
			val_correct += (pred == y_batch).sum().item()
			val_total += y_batch.size(0)
	val_acc = val_correct / val_total

	if verbose:
		print(f"DNN Val Accuracy: {val_acc:.4f}")

	if model_save_path:
		torch.save(model.state_dict(), model_save_path)

	test_acc = None
	if eval_on_test:
		with torch.no_grad():
			X_test = X_test.to(device)
			y_test = y_test.to(device)
			pred_test = model(X_test).argmax(dim=1)
			test_acc = (pred_test == y_test).sum().item() / y_test.size(0)
			print(f"DNN Test Accuracy: {test_acc:.4f}")

	return model, val_acc, test_acc




# ----------------------------
# XGBoost Training Function
# ----------------------------
# ----------------------------
# XGBoost Training Function
# ----------------------------
def train_xgboost(
	dictionary_path,
	num_estimators=100,
	max_depth=5,
	learning_rate=0.1,
	early_stopping_rounds=10,
	verbose=False,
	model_save_path=None,
	eval_on_test=False
):
	data = np.load(dictionary_path)
	X = np.stack([
		data['b0'], data['b0_mean'], data['b0_max'],
		data['FLAIR'], data['FLAIR_mean'], data['FLAIR_max']
	], axis=1).astype(np.float32)

	y = data['tissue_label'].astype(np.int64)

	# Remap labels to 0-based
	unique_labels = np.unique(y)
	label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
	y = np.vectorize(label_mapping.get)(y)
	num_classes = len(unique_labels)

	X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
	X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42)

	model = XGBClassifier(
		n_estimators=num_estimators,
		max_depth=max_depth,
		learning_rate=learning_rate,
		objective='multi:softprob',
		num_class=num_classes,
		use_label_encoder=False,
		tree_method='hist',
		predictor='gpu_predictor',
    device='gpu'
	)

	model.fit(
	X_train,
	y_train,
	eval_set=[(X_val, y_val)],
	verbose=verbose
)


	y_pred = model.predict(X_val)
	val_acc = accuracy_score(y_val, y_pred)
	if verbose:
		print(f"XGBoost Val Accuracy: {val_acc:.4f}")

	if model_save_path:
		joblib.dump(model, model_save_path)

	test_acc = None
	if eval_on_test:
		y_pred_test = model.predict(X_test)
		test_acc = accuracy_score(y_test, y_pred_test)
		print(f"XGBoost Test Accuracy: {test_acc:.4f}")

	return model, val_acc, test_acc



# ----------------------------
# Optuna Objective: DNN
# ----------------------------
dnn_val_acc_log = []

def dnn_objective(trial, dictionary_path):
	hidden_dim = trial.suggest_int("hidden_dim", 8, 24)
	dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
	lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
	depth = trial.suggest_int("depth", 1, 3)

	_, acc, _ = net_train(
		dictionary_path=dictionary_path,
		hidden_dim=hidden_dim,
		dropout_rate=dropout_rate,
		lr=lr,
		depth=depth,
		num_epochs=5,
		verbose=False
	)

	dnn_val_acc_log.append((trial.number, acc, hidden_dim, dropout_rate, lr, depth))
	return 1 - acc


# ----------------------------
# Optuna Objective: XGBoost
# ----------------------------
xgb_val_acc_log = []

def xgb_objective(trial, dictionary_path):
	num_estimators = trial.suggest_int("n_estimators", 50, 300)
	max_depth = trial.suggest_int("max_depth", 3, 10)
	learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)

	_, acc, _ = train_xgboost(
		dictionary_path=dictionary_path,
		num_estimators=num_estimators,
		max_depth=max_depth,
		learning_rate=learning_rate,
		verbose=False
	)

	# Save val acc
	xgb_val_acc_log.append((trial.number, acc, num_estimators, max_depth, learning_rate))
	return 1 - acc


if __name__ == "__main__":
	# ----------------------------
	# Run Optuna Tuning
	# ----------------------------
	dictionary_path = "/content/drive/MyDrive/IVIM_NeuroCovid/tissue_dictionary/training_dictionary.npz"


	print("Running DNN hyperparameter tuning with Optuna")
	dnn_objective_with_path = partial(dnn_objective, dictionary_path=dictionary_path)
	dnn_study = optuna.create_study(direction="minimize", study_name="DNN_Tuning")
	dnn_study.optimize(dnn_objective_with_path, n_trials=15)


	# Retrain and save best DNN model
	best_dnn_params = dnn_study.best_params
	dnn_model, val_acc_dnn, test_acc_dnn = net_train(
		dictionary_path=dictionary_path,
		hidden_dim=best_dnn_params["hidden_dim"],
		dropout_rate=best_dnn_params["dropout_rate"],
		lr=best_dnn_params["lr"],
		depth=best_dnn_params["depth"],
		num_epochs=10,
		model_save_path="best_dnn_model.pt",
		verbose=True,
		eval_on_test=True
	)
	print(f"DNN Final Test Accuracy: {test_acc_dnn:.4f}")

	print("Running XGBoost hyperparameter tuning with Optuna")
	xgb_objective_with_path = partial(xgb_objective, dictionary_path=dictionary_path)
	xgb_study = optuna.create_study(direction="minimize", study_name="XGB_Tuning")
	xgb_study.optimize(xgb_objective_with_path, n_trials=15)


	# Retrain and save best XGBoost model
	best_xgb_params = xgb_study.best_params
	xgb_model, val_acc_xgb, test_acc_xgb = train_xgboost(
		dictionary_path=dictionary_path,
		num_estimators=best_xgb_params["n_estimators"],
		max_depth=best_xgb_params["max_depth"],
		learning_rate=best_xgb_params["learning_rate"],
		model_save_path="best_xgb_model.pkl",
		verbose=True,
		eval_on_test=True
	)
	print(f"XGBoost Final Test Accuracy: {test_acc_xgb:.4f}")

	print("\n===== Final Model Performance Summary =====")
	print(f"DNN Best Config     : {best_dnn_params}")
	print(f"DNN Val Accuracy    : {val_acc_dnn:.4f}")
	print(f"DNN Test Accuracy   : {test_acc_dnn:.4f}")
	print()
	print(f"XGBoost Best Config : {best_xgb_params}")
	print(f"XGBoost Val Accuracy: {val_acc_xgb:.4f}")
	print(f"XGBoost Test Accuracy: {test_acc_xgb:.4f}")




		




