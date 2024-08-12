import librosa
import librosa.display
import librosa.beat
import sounddevice as sd
import warnings
import numpy as np
import matplotlib.pyplot as plt

import csv
import os
import librosa
from scipy import stats

import scipy
from scipy.spatial import distance
from math import ceil, floor

from sklearn.metrics import mean_squared_error
from math import sqrt

line_index = 0

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"
output_path = root_dir + "data/output/"
input_path = root_dir + "data/input/"

results = np.empty((900, 190))
results_norm = np.empty((902, 190))


# Exercise 2.1.2
def calculate_statistics(data):
	# print(f"\n->>>>DEBUG: {data.shape}")
	"""calculate and return the statistical measures for a given dataset.
	"""
	mean = np.mean(data)
	std_dev = np.std(data)
	skewness = stats.skew(data).reshape(())
	kurtosis = stats.kurtosis(data).reshape(())
	median = np.median(data)
	max_val = np.max(data)
	min_val = np.min(data)

	statistics = np.array([mean, std_dev, skewness, kurtosis, median, max_val, min_val], dtype=np.float64)
	statistics = statistics.reshape(1, 7)  # Reshape the array to 1 row by 7 columns

	# print(f"\nStatistics: {statistics}")
	return statistics


# Exercise 2.1.1
def extract_features_sample(sample):

	col_index = 0
	sample_stats = np.empty((190))

	#--- Extract Spectral Features
	#--- Extract Mel-frequency cepstral coefficients (MFCCs)
	mfcc = librosa.feature.mfcc(y = sample, n_mfcc=13)

	for value in mfcc:
		value_stats = calculate_statistics(value)
		sample_stats[col_index:col_index+7] = value_stats
		col_index += 7

	#--- Extract Spectral Centroid
	sc = librosa.feature.spectral_centroid(y = sample)
	sample_stats[col_index:col_index+7] = calculate_statistics(sc.transpose())
	col_index += 7

	#--- Extract Spectral Bandwidth
	sb = librosa.feature.spectral_bandwidth(y = sample)
	sample_stats[col_index:col_index+7] = calculate_statistics(sb.transpose())
	col_index += 7

	#--- Extract Spectral Contrast
	sct = librosa.feature.spectral_contrast(y = sample)

	for value in sct:
		sample_stats[col_index:col_index+7] = calculate_statistics(value)
		col_index += 7

	#--- Extract Spectral Flatness
	sf = librosa.feature.spectral_flatness(y = sample)
	sample_stats[col_index:col_index+7] = calculate_statistics(sf.transpose())
	col_index += 7

	#--- Extract Spectral Rolloff
	sr = librosa.feature.spectral_rolloff(y = sample)
	sample_stats[col_index:col_index+7] = calculate_statistics(sr.transpose())
	col_index += 7

	#--- Extract Temporal Features
	#--- Extract Fundamental Frequency 
	f0 = librosa.yin(sample, fmin=20, fmax=11025) # 20 Hz - 11025 Hz related to human audible range

	f0[f0 == 11025] = 0
	
	sample_stats[col_index:col_index+7] = calculate_statistics(f0)
	col_index += 7

	#--- Extract RMS Energy -7
	rms = librosa.feature.rms(y = sample)
	sample_stats[col_index:col_index+7] = calculate_statistics(rms.transpose())
	# print(results[line_index,col_index:col_index+7], end="\n\n")
	col_index += 7

	#--- Extract Zero Crossing Rate # -7
	zcr = librosa.feature.zero_crossing_rate(y = sample)
	sample_stats[col_index:col_index+7] = calculate_statistics(zcr.transpose())
	# print(results[line_index,col_index:col_index+7], end="\n\n")
	col_index += 7

	#--- Extract Tempo (Beats per minute) # 189
	# tempo, _ = librosa.beat.beat_track(y = sample)
	tempo = librosa.feature.tempo(y = sample)
	sample_stats[col_index] = tempo

	return sample_stats


# Exercise 2.1.3
def normalize_feature(low = 0, high = 1):

	for i in range(results_norm.shape[1]):
		results_norm[0, i] = results[:, i].min()
		results_norm[1, i] = results[:, i].max()
		# lower_bound + (feature - min) * (upper_bound - lower_bound) / (max - min)
		if results_norm[1, i] != results_norm[0, i]:
			results_norm[2:, i] = low + (results[:, i] - results_norm[0, i]) * (high - low) / (results_norm[1, i] - results_norm[0, i])
		else:
			results_norm[2:, i] = 0


def iterate_through_mp3s_extract_data(dataset_path, filename="results", low=0,  high=1, sr=22050, mono=True):

	global line_index

	for file_name in sorted(os.listdir(dataset_path)):
		print(f"Processing {file_name}")

		if file_name.endswith(".mp3"):
			file_path = os.path.join(dataset_path, file_name)
			sample, _ = librosa.load(file_path, sr=sr, mono=mono)
			# sample = np.pad(sample, (0, sample_size - len(sample)))
			results[line_index] = extract_features_sample(sample)

			line_index += 1

	# Normalize the features
	normalize_feature(low, high)
	# save the results
	save_results(filename)


def save_results(filename):

	global output_path
	np.savetxt(output_path + filename + ".csv", results, fmt="%f", delimiter=",")
	np.savetxt(output_path + filename + "_norm.csv", results_norm, fmt="%f", delimiter=",")


def open_csv(filename):
	return np.genfromtxt(filename, delimiter=",", dtype=None, encoding='utf-8')


def short_time_fourier_transform(signal, window_length=2048, hop_length=512):
	# number_of_windows = (signal.shape[0] - window_length) // hop_length + 1 - 3
	number_of_windows = signal.shape[0] // hop_length - 3
	print(f"Number of windows: {number_of_windows}")
	n_coefs = window_length // 2 + 1 if window_length % 2 == 0 else (window_length + 1) // 2

	stft = np.zeros((number_of_windows, n_coefs))

	for i in range(number_of_windows):
		# print(f"Processing window stft {i} ")
		if i == number_of_windows - 1:
			if window_length > signal[i*hop_length:].shape[0]:
				window_slice = signal[i*hop_length:]
				window_slice = np.pad(window_slice, (0, window_length - window_slice.shape[0]))
			else:
				window_slice = signal[i*hop_length:i*hop_length + window_length]
		else:
			window_slice = signal[i*hop_length:i*hop_length + window_length]
		
		windowed_signal = window_slice * np.hanning(window_length)
		stft_window = scipy.fft.rfft(windowed_signal)
		stft[i] = np.abs(stft_window)

	return stft


def spectral_centroid(signal, sr=22050, window_length=2048, hop_length=512):
	stft = short_time_fourier_transform(signal, window_length, hop_length)
	df = sr / window_length

	frequencies = np.arange(0, sr/2+1, df)

	spectral_centroid = np.zeros(stft.shape[0])

	for i in range(stft.shape[0]):
		stft_sum = np.sum(stft[i])
		if stft_sum == 0:
			spectral_centroid[i] = 0
		else:
			spectral_centroid[i] = np.sum(frequencies * stft[i]) / np.sum(stft[i])

	return spectral_centroid


def sc_comparison(dataset_path, sr=22050, mono=True):
	
	global line_index
	line_index = 0
	metricsSC = np.empty((900, 2))

	for file_name in sorted(os.listdir(dataset_path)):
		# print(f"Processing {file_name}")

		if file_name.endswith(".mp3"):
			file_path = os.path.join(dataset_path, file_name)
			sample, _ = librosa.load(file_path, sr=sr, mono=mono)

			librosa_sc = librosa.feature.spectral_centroid(y = sample)
			librosa_sc = librosa_sc.reshape(librosa_sc.shape[1])

			# librosa_sc = librosa_sc[2:]
			# our_sc = spectral_centroid(sample)
			# our_sc = our_spectral_centroid(sample)

			# print(f"librosa's spectral centroid shape: {librosa_sc.shape}, our's shape: {our_sc.shape}", end=" ")

			sc_size = min(librosa_sc.shape[0], our_sc.shape[0])
			librosa_sc = librosa_sc[:sc_size]
			our_sc = our_sc[:sc_size]

			# possivel sitio onde colocar
			librosa_sc = librosa_sc[2:]
			our_sc = spectral_centroid(sample)

			pearson_corr = np.corrcoef(librosa_sc, our_sc)
			# rmse = get_rmse(librosa_sc, our_sc)
			rmse = sqrt(mean_squared_error(librosa_sc, our_sc))

			metricsSC[line_index, 0] = pearson_corr[0, 1]
			metricsSC[line_index, 1] = rmse

			print(f"Pearson correlation: {pearson_corr[0, 1]}")
			line_index += 1

	np.savetxt(output_path + "metrics_sc.csv", metricsSC, fmt="%f", delimiter=",")


def our_spectral_centroid(signal, sr=22050, window_length=2048, hop_length=512):
	# calculate the number of windows
	n_windows = signal.shape[0] // hop_length - 3

	# padding if necessary
	padded_signal = np.zeros(window_length + ((n_windows) * hop_length))
	padded_signal[:signal.shape[0]] = signal

	# calculate the number of "frequency cells (???????)"
	# n_cells = (window_length // 2 + 1) / 2 * (sr / window_length)

	spectral_centroid = np.zeros(n_windows)

	for i in range(n_windows):
		# print(f"Processing window {i}")
		# get the windowed signal
		windowed_signal = padded_signal[i*hop_length:i*hop_length + window_length]
		windowed_signal = windowed_signal * np.hanning(window_length)

		# calculate the fft
		fft = np.fft.rfft(windowed_signal)
		fft = np.abs(fft)

		# calculate the frequencies
		frequencies = np.fft.fftfreq(window_length, 1/sr)
		frequencies = frequencies[:window_length // 2 + 1]

		# calculate the spectral centroid
		if np.sum(fft) == 0:
			spectral_centroid[i] = 0
		else:
			spectral_centroid[i] = np.sum(frequencies * fft) / np.sum(fft)
		# print(f"Spectral centroid: {spectral_centroid}")

	return spectral_centroid


# Exercise 3.1.1
def euclidean_distance(f1, f2):
	return distance.euclidean(f1, f2)

# Exercise 3.1.2
def manhattan_distance(f1, f2):
	return distance.cityblock(f1, f2)

# Exercise 3.1.3
def cosine_distance(f1, f2):
	return distance.cosine(f1, f2)

# Exercise 3.2.1
def norm_query_features(query_filename, low = 0, high = 1):

	# read query file
	query_sample, _ = librosa.load(query_filename, sr=22050, mono=True)
	
	# extract features' statistics
	# features = np.empty((190), dtype=np.float64)
	features = extract_features_sample(query_sample)

	# normalize query's statistics
	for i in range(results_norm.shape[1]):
		if results_norm[1, i] != results_norm[0, i]:
			features[i] = low + ((features[i] - results_norm[0, i]) * (high - low)) / (results_norm[1, i] - results_norm[0, i])
		else:
			features[i] = 0

	return features


def calculate_euclidean_distance(array1, array2):

	# Calculate the squared differences between corresponding elements of array1 and array2
	squared_differences = (array1 - array2) ** 2
	# Sum the squared differences
	sum_of_squared_differences = np.sum(squared_differences)
	# Take the square root of the sum to get the Euclidean distance
	distance = np.sqrt(sum_of_squared_differences)
	return distance


# Exercise 3.2.2
def save_similarity_matrix(normalized_ft_fname, query_norm):

	global output_path
	euclidean_matrix = np.zeros((results_norm.shape[0]-2)) # -2 because we have the min and max values in the first two rows
	manhattan_matrix = np.zeros((results_norm.shape[0]-2))
	cosine_matrix = np.zeros((results_norm.shape[0]-2))

	for i in range(2, results_norm.shape[0]):

		euclidean_matrix[i-2] = calculate_euclidean_distance(results_norm[i], query_norm)
		# euclidean_matrix[i-2] = euclidean_distance(results_norm[i], query_norm)
		manhattan_matrix[i-2] = manhattan_distance(results_norm[i], query_norm)
		cosine_matrix[i-2] = cosine_distance(results_norm[i], query_norm)

	np.savetxt(output_path + normalized_ft_fname + "_euclidean.csv", euclidean_matrix, fmt="%f", delimiter=",")
	np.savetxt(output_path + normalized_ft_fname + "_manhattan.csv", manhattan_matrix, fmt="%f", delimiter=",")
	np.savetxt(output_path + normalized_ft_fname + "_cosine.csv", cosine_matrix, fmt="%f", delimiter=",")


#Exercise 3.3
def create_similarity_ranking(normalized_ft_fname, input_file):

	euc_dist = open_csv(output_path + normalized_ft_fname + "_euclidean.csv")
	man_dist = open_csv(output_path + normalized_ft_fname + "_manhattan.csv")
	cos_dist = open_csv(output_path + normalized_ft_fname + "_cosine.csv")

	euc_ranking = np.argsort(euc_dist)
	man_ranking = np.argsort(man_dist)
	cos_ranking = np.argsort(cos_dist)

	# top 10 for each distance metric
	top_10_euc = euc_ranking[1:11]
	top_10_man = man_ranking[1:11]
	top_10_cos = cos_ranking[1:11]

	#open the file with the characteristics
	characteristics = open_csv(input_path + input_file)
	# print(characteristics)

	print("- Rankings:\n")
	# for each index in the top 10, print the name of the file
	print("Ranking: Euclidean-------------")
	top10 = []
	dists = []
	
	for i in top_10_euc:
		top10.append(str(characteristics[i+1][0]).strip("\""))
		dists.append(euc_dist[i])
	
	print(top10)
	print(f"dist = {dists}")

	top10 = []
	dists = []

	print("\nRanking: Manhattan-------------")
	for i in top_10_man:
		top10.append(str(characteristics[i+1][0]).strip("\""))
		dists.append(man_dist[i])
	
	print(top10)
	print(f"dist = {dists}")

	top10 = []
	dists = []

	print("\nRanking: Cosine-------------")
	for i in top_10_cos:
		top10.append(str(characteristics[i+1][0]).strip("\""))
		dists.append(cos_dist[i])

	print(top10)
	print(f"dist = {dists}")

	return top_10_euc, top_10_man, top_10_cos

def md_similarity_matrix(md_q_file, md_file, results_file="mdSim.csv"):
	
	query_md = open_csv(input_path + "other/" + md_q_file)[1:]			# read metadata of the query and ignore 1st row
	metadata = open_csv(input_path + "other/" + md_file)[1:]
 
	# columns to get: Artist, MoodsStrSplit and GenresStr, 1, 9 and 11 respectively
	metadata = metadata[:, [1, 9, 11]]			# get the columns with the metadata
	query_md = query_md[:, [1, 9, 11]]			# get the columns with the metadata

	metadata = np.char.replace(metadata, "\"", "")
	metadata = np.char.split(metadata, sep="; ")
	query_md = np.char.replace(query_md, "\"", "")
	query_md = np.char.split(query_md, sep="; ")

	similarity_matrix = np.zeros((metadata.shape[0]), dtype=np.int32)

	for i in range(metadata.shape[0]):
		similarity_matrix[i] += len(np.intersect1d(query_md[0, 0], metadata[i, 0]))
		similarity_matrix[i] += len(np.intersect1d(query_md[0, 1], metadata[i, 1]))
		similarity_matrix[i] += len(np.intersect1d(query_md[0, 2], metadata[i, 2]))
	
	np.savetxt(output_path + results_file, similarity_matrix, fmt="%d", delimiter=",")

	return similarity_matrix



def metadata_ranking(caracteristics_file, similarity_file = "mdSim.csv"):
	"""Exercise 4.1.1 \n
	Create a ranking of the metadata based on the similarity between the query and the metadata"""

	similarity_matrix = open_csv(output_path + similarity_file)
	characteristics = open_csv(input_path + caracteristics_file) #open the file with the characteristics

	ranking = np.argsort(similarity_matrix)[::-1][1:11] # ignore the first row, which has the best match since it's the query itself

	rank10 = []
	score = []

	print("\nRanking: Metadata-------------")
	for i in ranking:
		rank10.append(str(characteristics[i+1][0]).strip("\""))
		score.append(similarity_matrix[i])

	print(rank10)
	print(f"score = {score}")
	
	return ranking


def precision_calculation(metadata_ranking, euc_ranking, man_ranking, cos_ranking):
	"""Exercise 4.1.2 \n
	Calculate the precision  between the metadata ranking and the distance metrics ranking."""

	print()

	# count the number of equal ranks between the distances and the metadata similarity
	equal_ranks = len(np.intersect1d(metadata_ranking, euc_ranking))
	euc_precision = equal_ranks / 10 * 100
	print(f"Precision de: {euc_precision}")

	equal_ranks = len(np.intersect1d(metadata_ranking, man_ranking))
	man_precision = equal_ranks / 10 * 100
	print(f"Precision dm: {man_precision}")

	equal_ranks = len(np.intersect1d(metadata_ranking, cos_ranking))
	cos_precision = equal_ranks / 10 * 100
	print(f"Precision dc: {cos_precision}")

	return euc_precision, man_precision, cos_precision


def listen_top10(top10, chars_file, duration=5):
		play = True

		characteristics = open_csv(input_path + "other/" +  chars_file)

		print("\nListening to the top 10 songs:")
		for en, i in enumerate(top10):
			music = str(characteristics[i+1][0]).strip("\"")

			while play:
				print(f"music #{en+1}:")
				y, fs = librosa.load(input_path + "audio/" + music + ".mp3", duration=duration)
				sd.play(y, fs, blocking=True)

				if str(input()) != "x":
					break
	

def statistics(eval_file, title):
	"""Exercise 4.2."""
	evaluations = open_csv(input_path + eval_file)[1:] # 10 x 4 matrix with the evaluations of the 10 songs by 4 people

	# convert the evaluations to integers
	evaluations = evaluations.astype(np.int32)

	# calculate the mean and standard deviation of the evaluations for each document (line/song)
	docs_eval_mean = np.zeros((evaluations.shape[0]))
	docs_eval_std = np.zeros((evaluations.shape[0]))

	for i in range(evaluations.shape[0]): # for each song
		
		docs_eval_mean[i] = np.mean(evaluations[i])
		docs_eval_std[i] = np.std(evaluations[i])

	print(f"\n{title} -----------------\n")

	print(f"Docs eval mean: {docs_eval_mean}")
	print(f"Docs eval std: {docs_eval_std}\n")

	"""
 	Calcular a precisão das recomendações, dado pela percentagem de recomendações relevantes;
	Uma recomendação é relevante se o valor médio é no mínimo 2.5, caso contrário é considerada irrelevante.
  	"""

	# calculate the mean and standard deviation of the evaluations for each user (column)

	user_eval_mean = np.zeros((evaluations.shape[1]))
	user_eval_std = np.zeros((evaluations.shape[1]))

	for i in range(evaluations.shape[1]): # for each user
		
		user_eval_mean[i] = np.mean(evaluations[:,i])
		user_eval_std[i] = np.std(evaluations[:,i])

	print(f"Users eval mean: {user_eval_mean}")
	print(f"Users eval std: {user_eval_std}\n")

	# calculate the global mean and standard deviation of the evaluations
	global_eval_mean = np.mean(evaluations)
	global_eval_std = np.std(evaluations)

	print(f"Global eval mean: {global_eval_mean}")
	print(f"Global eval std: {global_eval_std}\n")

	precision = len(docs_eval_mean[docs_eval_mean >= 2.5]) / 10 * 100
	print(f"\nPrecision: {precision}")


	return

if __name__ == "__main__":

	# TODO: remove the appends

	extract = False
	plt.close('all')

	if extract:
		# #--- Load the audio files and extract their statistics
		iterate_through_mp3s_extract_data(input_path + "audio")
		sc_comparison(input_path + "audio")

		# extract the spectral centroid of the example
		sc_example, _ = librosa.load(input_path+"audio/MT0040033011.mp3", sr=22050, mono=True)
		np.savetxt(output_path + "query_sc.csv", spectral_centroid(sc_example), fmt="%f", delimiter=",")
		
		md_similarity_matrix("query_metadata.csv", "panda_dataset_taffc_metadata.csv")

	else:
		results = open_csv(output_path + "results.csv")
		results_norm = open_csv(output_path + "results_norm.csv")

		#-- Normalized query features
		query_features = norm_query_features(root_dir + "queries/MT0000414517.mp3")
		save_similarity_matrix("dist", query_features)
		top_10_euc, top_10_man, top_10_cos = create_similarity_ranking("dist", "other/panda_dataset_taffc_metadata.csv")

		md_ranking = metadata_ranking("other/panda_dataset_taffc_metadata.csv")
		precision_calculation(md_ranking, top_10_euc, top_10_man, top_10_cos)

		# print("Top 10 - Euclidean")
		# listen_top10(top_10_euc, "panda_dataset_taffc_metadata.csv", duration=10)
		# print("Ended")
		# input()

		# print("Top 10 - Manhattan")
		# listen_top10(top_10_man, "panda_dataset_taffc_metadata.csv", duration=10)
		# print("Ended")
		# input()
		
		# print("Top 10 - Cossine")
		# listen_top10(top_10_cos, "panda_dataset_taffc_metadata.csv", duration=10)
		# print("Ended")
		# input()

		# print("Top 10 - Metadata")
		# listen_top10(md_ranking, "panda_dataset_taffc_metadata.csv", duration=10)
		# print("Ended")
		# input()

		statistics("other/eval_euc.csv", "Euclidean")
		statistics("other/eval_man.csv", "Manhattan")
		statistics("other/eval_cos.csv", "Cosine")
		statistics("other/eval_md.csv", "Metadata")

#																												[Comentários gerais sobre a média]
# Euclidean
#	 João, Joel, Johnny, Tomás					# Batida?		# Instrumento?	# Género ?		# Emoções?		# Comentários [gerais]
# 01 5	   5     5       5		MT0003949060	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]	Muito parecida no ritmo, instrumentos e estilo
# 02 1	   3     1	     3		MT0004274911	[.][+][.][+]	[.][-][.][+]	[.][-][-][-]	[.][+][+][-] 	Idêntica na batida, com gênero, instrumentos e emoções distintas
# 03 1	   1     2	     2		MT0001515531	[.][-][-][-]	[-][-][-][-]	[-][-][-][-]	[-][-][.][-]	Muito distinta em todos os aspetos
# 04 1	   2     2	     2		MT0003900455	[.][-][+][+]	[.][+][+][+]	[.][-][.][-]	[.][+][.][-]	Semelhança nos instrumentos, nas emoções e na melodia
# 05 3	   2     3	     2		MT0009897495	[.][-][+][+]	[+][-][+][+]	[.][-][.][-]	[.][+][+][-]	Semelhança na instrumentação
# 06 4	   4     5       4		MT0000040632	[.][+][+][+]	[+][+][+][+]	[+][+][+][-]	[.][+][+][-]	Uso de instrumentos semelhantes e com um gênero parecido
# 07 1	   2     3	     1		MT0004032071	[.][-][.][-]	[+][+][+][-]	[.][-][-][-]	[+][-][+][-]	
# 08 3	   2     3       2		MT0005469880	[.][-][.][-]	[+][+][+][-]	[.][-][.][-]	[.][-][+][-]	
# 09 3     2     3       2		MT0007043936	[.][-][-][-]	[+][+][+][+]	[.][-][.][-]	[+][-][.][-]	
# 10 2     2     2       1		MT0034005433	[.][-][-][-]	[+][+][+][-]	[.][-][-][-]	[.][-][-][-]	

# Manhattan	
#	 João, Joel, Johnny, Tomás					# Batida?		# Instrumento?	# Género ?		# Emoções?		# Comentários [gerais]
# 01 5	   5     5       5		MT0003949060	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]
# 02 1	   3     1	     3		MT0004274911	[.][+][.][+]	[.][-][.][+]	[.][-][-][-]	[.][+][+][-]
# 03 4	   4     5       4		MT0000040632	[.][+][+][+]	[+][+][+][+]	[+][+][+][-]	[.][+][+][-]
# 04 1	   2     1       2		MT0000218346	[.][-][-][-]	[+][+][.][+]	[.][-][-][-]	[-][-][-][-]
# 05 1	   2     2	     2		MT0003900455	[.][-][+][+]	[.][+][+][+]	[.][-][.][-]	[.][+][.][-]
# 06 3	   2     3       2		MT0005469880	[.][-][.][-]	[+][+][+][-]	[.][-][.][-]	[.][-][+][-]
# 07 3	   3     2       2		MT0008401073	[.][+][+][+]	[+][+][+][+]	[-][-][.][-]	[.][-][.][-]
# 08 1	   1     2	     2		MT0001515531	[.][-][-][-]	[-][-][-][-]	[-][-][-][-]	[-][-][.][-]
# 09 1	   2     3       1		MT0001624303	[.][-][-][-]	[.][-][-][-]	[-][-][-][-]	[-][+][-][-]
# 10 3	   2     3	     2		MT0009897495	[.][-][+][+]	[+][-][+][+]	[.][-][.][-]	[.][+][+][-]

# Cosine
#	 João, Joel, Johnny, Tomás					# Batida?		# Instrumento?	# Género ?		# Emoções?		# Comentários [gerais]
# 01 1	   3     1	     3		MT0004274911	[.][+][.][+]	[.][-][.][+]	[.][-][-][-]	[.][+][+][-]
# 02 5	   5     5       5		MT0003949060	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]
# 03 1	   1     2	     2		MT0001515531	[.][-][-][-]	[-][-][-][-]	[-][-][-][-]	[-][-][.][-]
# 04 1	   2     3	     1		MT0002634024	[.][-][-][-]	[.][+][-][-]	[-][-][-][-]	[-][-][-][-]
# 05 1	   2     2	     2		MT0003900455	[.][-][+][+]	[.][+][+][+]	[.][-][.][-]	[.][+][.][-]
# 06 3	   2     3	     2		MT0009897495	[.][-][+][+]	[+][-][+][+]	[.][-][.][-]	[.][+][+][-]
# 07 2	   3     2	     3		MT0001942272	[.][+][+][+]	[-][-][-][+]	[-][-][-][-]	[-][+][-][-]
# 08 1	   2     3	     1		MT0004032071	[.][-][.][-]	[+][+][+][-]	[.][-][-][-]	[+][-][+][-]
# 09 4	   4     5       4		MT0000040632	[.][+][+][+]	[+][+][+][+]	[+][+][+][-]	[.][+][+][-]
# 10 1	   2     2	     1		MT0005752234	[.][-][+][-]	[.][+][-][-]	[-][-][-][-]	[-][-][-][-]
	
# Metadata	
#	 João, Joel, Johnny, Tomás					# Batida?		# Instrumento?	# Género ?		# Emoções?		# Comentários [gerais]
# 01 5	   5     5       5		MT0027048677	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]
# 02 5	   5     5       5		MT0010489498	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]
# 03 3	   3     5       4		MT0010487769	[+][+][+][+]	[+][+][+][+]	[.][+][+][+]	[.][-][+][+]
# 04 4	   4     4       3		MT0033397838	[+][+][+][+]	[+][+][+][+]	[.][+][+][+]	[+][+][+][+]
# 05 5	   5     5       5		MT0003949060	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]
# 06 4	   5     4       4		MT0012331779	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]	[+][+][+][+]
# 07 4	   4     5       4		MT0000040632	[.][+][+][+]	[+][+][+][+]	[+][+][+][-]	[.][+][+][-]
# 08 4	   4     5       4		MT0002222957	[.][+][+][+]	[+][+][+][+]	[+][-][+][+]	[.][-][+][+]
# 09 3	   5     3       2		MT0008222676	[+][+][-][+]	[+][+][-][-]	[+][+][+][-]	[+][+][.][-]
# 10 1	   4     2       1		MT0010900969	[.][+][-][-]	[.][+][-][-]	[-][-][-][-]	[-][-][.][-]
