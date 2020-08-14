import csv

for i in range(1, 1):
	with open('log_' + str(i) + '.csv') as file:
		string = file.readlines()[0]
		print(string)