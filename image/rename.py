import os



files = [file for file in os.listdir() if file.endswith('.png')]
print(len(files))

for i , file  in enumerate(files):
    n_name = '{}.jpg'.format(i+1)
    os.rename(file, n_name)
    print(n_name)
