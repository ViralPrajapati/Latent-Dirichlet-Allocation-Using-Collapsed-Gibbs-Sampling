# Latent Dirichlet Allocation

LDA is a generative topic modeling approach which takes into consideration the 
latent structure that underlies within the set of words. With the use of Gibbs 
Sampling it is able to generate a sophisticated and statistically stable model 
that can make pretty good classifications.

Structure of the files:
LDA.py

Command to run the files:

LDA.py
output = Graph of Accuracy vs Datasize

PS: The code uses the "os.path.dirname(os.path.abspath(__file__))" command to get the current directory location to read the dataset. If in case of any error relating to file or directory not found please add the path to the dataset at line 278.

The code assumes that dataset is present in same working directory and has seperate folder for "20newsgroups" and "artificial" data. 
