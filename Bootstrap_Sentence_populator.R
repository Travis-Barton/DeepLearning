# Bootstrap sampling of individual words in each group of sentences seperated by class. 
# This way we get more data for small sizes


### 1) Turn posts into one hot encodings with data.
Word_Samples = function(titles, tags, stopwords){
  total_words = list()
  total_lengths = list()
  for(tag in unique(tags)){
    index = which(tags == tag)
    total_words[[tag]] = unlist(lapply(titles[index], strsplit, split = ' ', fixed = TRUE))
    bad_words_index = which(total_words[[tag]] %in% stopwords)
    total_words[[tag]] = total_words[[tag]][-bad_words_index]
    total_lengths[[tag]] = lengths(unlist(lapply(titles[index], strsplit, split = ' ', fixed = TRUE), recursive = F))
  }
  return(list(word_bank = total_words, length_bank = total_lengths))
}
### 2) Seperate them by class.

### 3) Label each '1' in each vector with a vector and a placeholder.

### 4) Make a vector of lengths of words.

### 5) Sample each word randomly. The size of each word is also sampled from the 
###    vector made in 4). 

### 6) Profit! (jk but really then you have new data and can test)

### 7) Functionalize it for package.


########################## Sandbox Area ##########################
# Lets get some test data going
library(readr) # Reading in CSV
library(dplyr) # For piping
library(text2vec) # For Glove Model
library(tm) # For stemming
library(textclean) # For certain character removles 
dat <- read.csv("~/Documents/GitHub/Redditbot/askscience_Data.csv")
titles = as.character(dat$Title)
titles = unlist(Pretreatment(titles))
stopwords = StopWordMaker(titles, cutoff = 30)
emb_mat = t(Embedding_Matrix(titles, 5L, stopwords, 10L, 50))



