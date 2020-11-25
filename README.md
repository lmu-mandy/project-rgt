# project-rgt
project-rgt created by GitHub Classroom

To read in the zipped preprocessed data file, use these lines of code.

url = 'https://github.com/lmu-mandy/project-rgt/blob/main/preprocessed_data.csv.zip?raw=true'  
df = pd.read_csv(url, compression='zip', header=0, sep=',', quotechar='"', index_col=0)
