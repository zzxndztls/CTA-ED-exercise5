---
title: "CTA-ED Exercise 5: Unsupervised learning (topic models)"
author: "Zhitong Chen,Ka Hei Leung"
date: "17/03/2026"
output: pdf_document
---
## Introduction

The hands-on exercise for this week focuses on: 1) estimating a topic model ; 2) interpreting and visualizing results.
Remember that you will need to: 1) comment your code and 2) write out the interpretation of your results.

You will learn how to:

* Generate document-term-matrices in format appropriate for topic modelling
* Estimate a topic model using the `quanteda` and `topicmodels` package
* Visualize results
* Reverse engineer a test of model accuracy
* Run some validation tests

## Setup 

Before proceeding, we'll load the packages we will need for this tutorial.


``` r
library(tidyverse) # loads dplyr, ggplot2, and others
library(stringr) # to handle text elements
library(tidytext) # includes set of functions useful for manipulating text
library(topicmodels) # to estimate topic models
library(gutenbergr) # to get text data
library(scales)
library(tm)
library(ggthemes) # to make your plots look nice
library(readr)
library(quanteda)
library(quanteda.textmodels)
```

You may need to install the preText package if you haven't done so yet. For that you will need to run the next code chunk (it is currently set to 'eval=F', which tells R 'do not execute this code chunk').
That package is not readily available on through RStudio directly. It needs to be downloaded from the Github repository set up by its creater Matthew J Denny. We can do that using the command install_github(). This command is part of the 'devtools' package, which you will need to install as well (if you haven't done so already). The devtools package is directly available through R so it can be installed with the usual command install_packages. 


``` r
#install_package(devtools)
devtools::install_github("matthewjdenny/preText")
library(preText)
```


# Data collection
We'll be using data from Alexis de Tocqueville's "Democracy in America." 

We have already downloaded some data for you, but we also included the code to download it yourself (it is currently set to 'eval=F' so it won't run unless you remove the eval=F argument or you run the chunk directly. 

The code downloads these data, both Volume 1 and Volume 2, and combine them into one data frame. For this, we'll be using the <tt>gutenbergr</tt> package, which allows the user to download text data from over 60,000 out-of-copyright books. The ID for each book appears in the url for the book selected after a search on [https://www.gutenberg.org/ebooks/](https://www.gutenberg.org/ebooks/).

This example is adapted by [Text Mining with R: A Tidy Approach](https://www.tidytextmining.com/) by Julia Silge and David Robinson.

![](data/topicmodels/gutenberg.gif){width=100%}

Here, we see that Volume of Tocqueville's "Democracy in America" is stored as "815". A separate search reveals that Volume 2 is stored as "816".


``` r
tocq <- gutenberg_download(c(815, 816), 
                            meta_fields = "author")
```

Or we can read the dataset we already downloaded for you in the following way:


``` r
tocq  <- readRDS(gzcon(url("https://github.com/cjbarrie/CTA-ED/blob/main/data/topicmodels/tocq.RDS?raw=true")))
```

Once we have read in these data, we convert it into a different data shape: the document-term-matrix. We also create a new columns, which we call "booknumber" that recordss whether the term in question is from Volume 1 or Volume 2. To convert from tidy into "DocumentTermMatrix" format we can first use `unnest_tokens()` as we have done in past exercises, remove stop words, and then use the `cast_dtm()` function to convert into a "DocumentTermMatrix" object.


``` r
tocq_words <- tocq %>%
  mutate(booknumber = ifelse(gutenberg_id==815, "DiA1", "DiA2")) %>%
  unnest_tokens(word, text) %>%
  filter(!is.na(word)) %>%
  count(booknumber, word, sort = TRUE) %>%
  ungroup() %>%
  anti_join(stop_words)
```

```
## Joining with `by = join_by(word)`
```

``` r
tocq_dtm <- tocq_words %>%
  cast_dtm(booknumber, word, n)

tm::inspect(tocq_dtm)
```

```
## <<DocumentTermMatrix (documents: 2, terms: 12092)>>
## Non-/sparse entries: 17581/6603
## Sparsity           : 27%
## Maximal term length: 18
## Weighting          : term frequency (tf)
## Sample             :
##       Terms
## Docs   country democratic government laws nations people power society time
##   DiA1     357        212        556  397     233    516   543     290  311
##   DiA2     167        561        162  133     313    360   263     241  309
##       Terms
## Docs   united
##   DiA1    554
##   DiA2    227
```

We see here that the data are now stored as a "DocumentTermMatrix." In this format, the matrix records the term (as equivalent of a column) and the document (as equivalent of row), and the number of times the term appears in the given document. Many terms will not appear in the document, meaning that the matrix will be stored as "sparse," meaning there will be a preponderance of zeroes. Here, since we are looking only at two documents that both come from a single volume set, the sparsity is relatively low (only 27%). In most applications, the sparsity will be a lot higher, approaching 99% or more.

Estimating our topic model is then relatively simple. All we need to do if specify how many topics that we want to search for, and we can also set our seed, which is needed to reproduce the same results each time (as the model is a generative probabilistic one, meaning different random iterations will produce different results).


``` r
tocq_lda <- LDA(tocq_dtm, k = 10, control = list(seed = 1234))
```

After this we can extract the per-topic-per-word probabilities, called "β" from the model:


``` r
tocq_topics <- tidy(tocq_lda, matrix = "beta")

head(tocq_topics, n = 10)
```

```
## # A tibble: 10 x 3
##    topic term          beta
##    <int> <chr>        <dbl>
##  1     1 democratic 0.00855
##  2     2 democratic 0.0115 
##  3     3 democratic 0.00444
##  4     4 democratic 0.0193 
##  5     5 democratic 0.00254
##  6     6 democratic 0.00866
##  7     7 democratic 0.00165
##  8     8 democratic 0.0108 
##  9     9 democratic 0.00276
## 10    10 democratic 0.00334
```

We now have data stored as one topic-per-term-per-row. The betas listed here represent the probability that the given term belongs to a given topic. So, here, we see that the term "democratic" is most likely to belong to topic 4. Strictly, this probability represents the probability that the term is generated from the topic in question.

We can then plots the top terms, in terms of beta, for each topic as follows:


``` r
tocq_top_terms <- tocq_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

tocq_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free", ncol = 4) +
  scale_y_reordered() +
  theme_tufte(base_family = "Helvetica")
```

![](unsupervised-topicmodels_files/figure-latex/unnamed-chunk-8-1.pdf)<!-- --> 

But how do we actually evaluate these topics? Here, the topics all seem pretty similar. 

## Evaluating topic model

Well, one way to evaluate the performance of unspervised forms of classification is by testing our model on an outcome that is already known. 

Here, two topics that are most obvious are the 'topics' Volume 1 and Volume 2 of Tocqueville's "Democracy in America." Volume 1 of Tocqueville's work deals more obviously with abstract constitutional ideas and questions of race; Volume 2 focuses on more esoteric aspects of American society. Listen an "In Our Time" episode with Melvyn Bragg discussing Democracy in America [here](https://www.bbc.co.uk/programmes/b09vyw0x).

Given these differences in focus, we might think that a generative model could accurately assign to topic (i.e., Volume) with some accuracy.

### Plot relative word frequencies

First let's have a look and see whether there really are words obviously distinguishing the two Volumes. 


``` r
tidy_tocq <- tocq %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words)
```

```
## Joining with `by = join_by(word)`
```

``` r
## Count most common words in both
tidy_tocq %>%
  count(word, sort = TRUE)
```

```
## # A tibble: 12,092 x 2
##    word           n
##    <chr>      <int>
##  1 people       876
##  2 power        806
##  3 united       781
##  4 democratic   773
##  5 government   718
##  6 time         620
##  7 nations      546
##  8 society      531
##  9 laws         530
## 10 country      524
## # i 12,082 more rows
```

``` r
bookfreq <- tidy_tocq %>%
  mutate(booknumber = ifelse(gutenberg_id==815, "DiA1", "DiA2")) %>%
  mutate(word = str_extract(word, "[a-z']+")) %>%
  count(booknumber, word) %>%
  group_by(booknumber) %>%
  mutate(proportion = n / sum(n)) %>% 
  select(-n) %>% 
  spread(booknumber, proportion)

ggplot(bookfreq, aes(x = DiA1, y = DiA2, color = abs(DiA1 - DiA2))) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  scale_color_gradient(limits = c(0, 0.001), low = "darkslategray4", high = "gray75") +
  theme_tufte(base_family = "Helvetica") +
  theme(legend.position="none", 
        strip.background = element_blank(), 
        strip.text.x = element_blank()) +
  labs(x = "Tocqueville DiA 2", y = "Tocqueville DiA 1") +
  coord_equal()
```

```
## Warning: Removed 6173 rows containing missing values or values outside the scale range
## (`geom_point()`).
```

```
## Warning: Removed 6174 rows containing missing values or values outside the scale range
## (`geom_text()`).
```

![](unsupervised-topicmodels_files/figure-latex/unnamed-chunk-9-1.pdf)<!-- --> 

We see that there do seem to be some marked distinguishing characteristics. In the plot above, for example, we see that more abstract notions of state systems appear with greater frequency in Volume 1 while Volume 2 seems to contain words specific to America (e.g., "north" and "south") with greater frequency. The way to read the above plot is that words positioned further away from the diagonal line appear with greater frequency in one volume versus the other.


### Split into chapter documents

In the below, we first separate the volumes into chapters, then we repeat the same procedure as above. The only difference now is that instead of two documents representing the two full volumes of Tocqueville's work, we now have 132 documents, each representing an individual chapter. Notice now that the sparsity is much increased: around 96%. 


``` r
tocq <- tocq %>%
  filter(!is.na(text))

# Divide into documents, each representing one chapter
tocq_chapter <- tocq %>%
  mutate(booknumber = ifelse(gutenberg_id==815, "DiA1", "DiA2")) %>%
  group_by(booknumber) %>%
  mutate(chapter = cumsum(str_detect(text, regex("^chapter ", ignore_case = TRUE)))) %>%
  ungroup() %>%
  filter(chapter > 0) %>%
  unite(document, booknumber, chapter)

# Split into words
tocq_chapter_word <- tocq_chapter %>%
  unnest_tokens(word, text)

# Find document-word counts
tocq_word_counts <- tocq_chapter_word %>%
  anti_join(stop_words) %>%
  count(document, word, sort = TRUE) %>%
  ungroup()
```

```
## Joining with `by = join_by(word)`
```

``` r
tocq_word_counts
```

```
## # A tibble: 69,781 x 3
##    document word             n
##    <chr>    <chr>        <int>
##  1 DiA2_76  united          88
##  2 DiA2_60  honor           70
##  3 DiA1_52  union           66
##  4 DiA2_76  president       60
##  5 DiA2_76  law             59
##  6 DiA1_42  jury            57
##  7 DiA2_76  time            50
##  8 DiA1_11  township        49
##  9 DiA1_21  federal         48
## 10 DiA2_76  constitution    48
## # i 69,771 more rows
```

``` r
# Cast into DTM format for LDA analysis

tocq_chapters_dtm <- tocq_word_counts %>%
  cast_dtm(document, word, n)

tm::inspect(tocq_chapters_dtm)
```

```
## <<DocumentTermMatrix (documents: 132, terms: 11898)>>
## Non-/sparse entries: 69781/1500755
## Sparsity           : 96%
## Maximal term length: 18
## Weighting          : term frequency (tf)
## Sample             :
##          Terms
## Docs      country democratic government laws nations people power public time
##   DiA1_11      10          0         23   19       7     13    19     15    6
##   DiA1_13      13          5         34    9      12     17    37     15    6
##   DiA1_20       9          0         25   13       2     14    32     13   10
##   DiA1_21       4          0         20   29       6     12    20      5    5
##   DiA1_23      10          0         35    9      24     20    13      4    8
##   DiA1_31       7         12         10   13       4     30    18     31    6
##   DiA1_32      10         14         25    6       9     25    11     43    8
##   DiA1_47      12          2          5    3       3      6     8      0    3
##   DiA1_56      12          0          3    7      19      3     8      3   22
##   DiA2_76      11         10         24   39      12     31    27     27   50
##          Terms
## Docs      united
##   DiA1_11     13
##   DiA1_13     19
##   DiA1_20     21
##   DiA1_21     23
##   DiA1_23     15
##   DiA1_31     11
##   DiA1_32     14
##   DiA1_47      8
##   DiA1_56     25
##   DiA2_76     88
```

We then re-estimate the topic model with this new DocumentTermMatrix object, specifying k equal to 2. This will enable us to evaluate whether a topic model is able to generatively assign to volume with accuracy.


``` r
tocq_chapters_lda <- LDA(tocq_chapters_dtm, k = 2, control = list(seed = 1234))
```

After this, it is worth looking at another output of the latent dirichlet allocation procedure. The γ probability represents the per-document-per-topic probability or, in other words, the probability that a given document (here: chapter) belongs to a particular topic (and here, we are assuming these topics represent volumes).

The gamma values are therefore the estimated proportion of words within a given chapter allocated to a given volume. 


``` r
tocq_chapters_gamma <- tidy(tocq_chapters_lda, matrix = "gamma")
tocq_chapters_gamma
```

```
## # A tibble: 264 x 3
##    document topic     gamma
##    <chr>    <int>     <dbl>
##  1 DiA2_76      1 0.551    
##  2 DiA2_60      1 1.000    
##  3 DiA1_52      1 0.0000464
##  4 DiA1_42      1 0.0000746
##  5 DiA1_11      1 0.0000382
##  6 DiA1_21      1 0.0000437
##  7 DiA1_20      1 0.0000425
##  8 DiA1_28      1 0.249    
##  9 DiA1_50      1 0.0000477
## 10 DiA1_22      1 0.0000466
## # i 254 more rows
```

### Examine consensus

Now that we have these topic probabilities, we can see how well our unsupervised learning did at distinguishing the two volumes generatively just from the words contained in each chapter.


``` r
# First separate the document name into title and chapter

tocq_chapters_gamma <- tocq_chapters_gamma %>%
  separate(document, c("title", "chapter"), sep = "_", convert = TRUE)

tocq_chapter_classifications <- tocq_chapters_gamma %>%
  group_by(title, chapter) %>%
  top_n(1, gamma) %>%
  ungroup()

tocq_book_topics <- tocq_chapter_classifications %>%
  count(title, topic) %>%
  group_by(title) %>%
  top_n(1, n) %>%
  ungroup() %>%
  transmute(consensus = title, topic)

tocq_chapter_classifications %>%
  inner_join(tocq_book_topics, by = "topic") %>%
  filter(title != consensus)
```

```
## # A tibble: 15 x 5
##    title chapter topic gamma consensus
##    <chr>   <int> <int> <dbl> <chr>    
##  1 DiA1       45     1 0.762 DiA2     
##  2 DiA1        5     1 0.504 DiA2     
##  3 DiA1       33     1 0.570 DiA2     
##  4 DiA1       34     1 0.626 DiA2     
##  5 DiA1       41     1 0.512 DiA2     
##  6 DiA1       44     1 0.765 DiA2     
##  7 DiA1        8     1 0.791 DiA2     
##  8 DiA1        4     1 0.717 DiA2     
##  9 DiA1       35     1 0.576 DiA2     
## 10 DiA1       39     1 0.577 DiA2     
## 11 DiA1        7     1 0.687 DiA2     
## 12 DiA1       29     1 0.983 DiA2     
## 13 DiA1        6     1 0.707 DiA2     
## 14 DiA2       27     2 0.654 DiA1     
## 15 DiA2       21     2 0.510 DiA1
```

``` r
# Look document-word pairs were to see which words in each documents were assigned
# to a given topic

assignments <- augment(tocq_chapters_lda, data = tocq_chapters_dtm)
assignments
```

```
## # A tibble: 69,781 x 4
##    document term   count .topic
##    <chr>    <chr>  <dbl>  <dbl>
##  1 DiA2_76  united    88      2
##  2 DiA2_60  united     6      1
##  3 DiA1_52  united    11      2
##  4 DiA1_42  united     7      2
##  5 DiA1_11  united    13      2
##  6 DiA1_21  united    23      2
##  7 DiA1_20  united    21      2
##  8 DiA1_28  united    14      2
##  9 DiA1_50  united     5      2
## 10 DiA1_22  united     8      2
## # i 69,771 more rows
```

``` r
assignments <- assignments %>%
  separate(document, c("title", "chapter"), sep = "_", convert = TRUE) %>%
  inner_join(tocq_book_topics, by = c(".topic" = "topic"))

assignments %>%
  count(title, consensus, wt = count) %>%
  group_by(title) %>%
  mutate(percent = n / sum(n)) %>%
  ggplot(aes(consensus, title, fill = percent)) +
  geom_tile() +
  scale_fill_gradient2(high = "red", label = percent_format()) +
  geom_text(aes(x = consensus, y = title, label = scales::percent(percent))) +
  theme_tufte(base_family = "Helvetica") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.grid = element_blank()) +
  labs(x = "Book words assigned to",
       y = "Book words came from",
       fill = "% of assignments")
```

![](unsupervised-topicmodels_files/figure-latex/unnamed-chunk-13-1.pdf)<!-- --> 

Not bad! We see that the model estimated with accuracy 91% of chapters in Volume 2 and 79% of chapters in Volume 1

## Validation

In the articles by @ying_topics_2021 and @denny_text_2018 from this and previous weeks, we read about potential validation techniques. 

In this section, we'll be using the `preText` package mentioned in @denny_text_2018 to see the impact of different pre-processing choices on our text. Here, I am adapting from a [tutorial](http://www.mjdenny.com/getting_started_with_preText.html) by Matthew Denny.

First we need to reformat our text into a `quanteda` corpus object. 


``` r
# load in corpus of Tocequeville text data.
corp <- corpus(tocq, text_field = "text")
# use first 10 documents for example
documents <- corp[sample(1:30000,1000)]
# take a look at the document names
print(names(documents[1:10]))
```

```
##  [1] "text22709" "text19967" "text1083"  "text9455"  "text15066" "text28819"
##  [7] "text6979"  "text23452" "text21076" "text3987"
```
And now we are ready to preprocess in different ways. Here, we are including n-grams so we are preprocessing the text in 128 different ways. This takes about ten minutes to run on a machine with 8GB RAM. 


``` r
preprocessed_documents <- factorial_preprocessing(
    documents,
    use_ngrams = TRUE,
    infrequent_term_threshold = 0.2,
    verbose = FALSE)
```

We can then get the results of our pre-processing, comparing the distance between documents that have been processed in different ways. 



``` r
preText_results <- preText(
    preprocessed_documents,
    dataset_name = "Tocqueville text",
    distance_method = "cosine",
    num_comparisons = 20,
    verbose = FALSE)
```

And we can plot these accordingly. 


``` r
preText_score_plot(preText_results)
```

![](data/topicmodels/pretext_results.png){width=100%}

## Exercises

1. Choose another book or set of books from Project Gutenberg

``` r
#load packages required for the tidytext workflow and modeling
library(tidyverse)    # data wrangling
library(tidytext)     # tidy text processing
library(gutenbergr)   # download Project Gutenberg texts
library(topicmodels)  # LDA implementation
library(preText)      # preprocessing robustness checks 
```

```
## preText: Diagnostics to Assess the Effects of Text Preprocessing Decisions
## Version 0.7.2 created on 2021-07-25.
## copyright (c) 2021, Matthew J. Denny, Georgetown University
##                     Arthur Spirling, NYU
## Type vignette('getting_started_with_preText') to get started.
## Development website: https://github.com/matthewjdenny/preText
```

``` r
# set a seed so that the results are reproducible
set.seed(1234)
```


``` r
# Download two Jane Austen novels by Gutenberg ID
pp <- gutenberg_download(1342)   # Pride and Prejudice
```

```
## Using mirror https://aleph.pglaf.org.
```

``` r
ss <- gutenberg_download(161)    # Sense and Sensibility

# Add a book label that we'll use as the "true" label later
pp <- pp %>% mutate(book = "Pride and Prejudice")
ss <- ss %>% mutate(book = "Sense and Sensibility")

# Combine the two books into one tibble
books <- bind_rows(pp, ss)

# Create chapter identifiers by detecting lines that start with "Chapter"
books <- books %>%
  mutate(chapter = cumsum(str_detect(text, regex("^chapter", ignore_case = TRUE)))) %>%
  filter(chapter > 0)  # drop front-matter before first chapter
```


``` r
# Tokenize into one word per row using tidytext's unnest_tokens 
tidy_books <- books %>%
  select(book, chapter, text) %>%
  unnest_tokens(word, text)

# Remove stopwords using the standard stop_words dataset
data("stop_words")
tidy_books <- tidy_books %>%
  anti_join(stop_words, by = "word")

# Count term frequency per document (document = paste(book, chapter))
dtm_counts <- tidy_books %>%
  count(document = paste(book, chapter), word, sort = FALSE)

# Cast to DocumentTermMatrix required by LDA
book_dtm <- dtm_counts %>%
  cast_dtm(document = document, term = word, value = n)

# Print DTM dimensions as a quick check
book_dtm
```

```
## <<DocumentTermMatrix (documents: 112, terms: 8151)>>
## Non-/sparse entries: 51529/861383
## Sparsity           : 94%
## Maximal term length: 17
## Weighting          : term frequency (tf)
```
2. Run your own topic model on these books, changing the k of topics, and evaluating accuracy.

Topic modelling (LDA)

``` r
library(topicmodels)

# Fit LDA model with k = 2 topics
lda_model <- LDA(book_dtm, k = 2, control = list(seed = 1234))

# Extract topic-word probabilities (beta)
beta <- tidy(lda_model, matrix = "beta")

# Display top 10 terms per topic
top_terms <- beta %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup()

top_terms
```

```
## # A tibble: 20 x 3
##    topic term         beta
##    <int> <chr>       <dbl>
##  1     1 elizabeth 0.0157 
##  2     1 darcy     0.00983
##  3     1 bennet    0.00783
##  4     1 miss      0.00765
##  5     1 jane      0.00699
##  6     1 bingley   0.00683
##  7     1 time      0.00527
##  8     1 lady      0.00499
##  9     1 sister    0.00490
## 10     1 dear      0.00427
## 11     2 elinor    0.0170 
## 12     2 marianne  0.0134 
## 13     2 time      0.00641
## 14     2 dashwood  0.00631
## 15     2 mother    0.00614
## 16     2 sister    0.00605
## 17     2 edward    0.00598
## 18     2 miss      0.00555
## 19     2 jennings  0.00544
## 20     2 colonel   0.00514
```

Document classification

``` r
# Extract document-topic probabilities (gamma)
gamma <- tidy(lda_model, matrix = "gamma")

# Separate document into book and chapter
gamma <- gamma %>%
  separate(document, into = c("book", "chapter"), sep = " ", extra = "merge")

# Assign each document to the most likely topic
doc_class <- gamma %>%
  group_by(book, chapter) %>%
  slice_max(gamma, n = 1) %>%
  ungroup()

# Map topics to books using majority rule
topic_map <- doc_class %>%
  count(topic, book) %>%
  group_by(topic) %>%
  slice_max(n, n = 1)

# Calculate accuracy
predictions <- doc_class %>%
  left_join(topic_map, by = "topic", suffix = c("", "_pred")) %>%
  mutate(correct = book == book_pred)

accuracy <- mean(predictions$correct)

accuracy
```

```
## [1] 1
```
3. Validate different pre-processing techniques using `preText` on the new book(s) of your choice. 


``` r
library(preText)

# Use a subset of the data to reduce computational cost
texts_small <- books$text[1:1000]

# Run factorial preprocessing WITHOUT n-grams to avoid memory issues
fp <- factorial_preprocessing(
  texts_small,
  use_ngrams = FALSE,
  parallel = FALSE,

)
```

```
## Preprocessing 1000 documents 64 different ways...
## Currently working on combination 1 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,154 sparse dfm.
```

```
## Removing 1087 of 1154 total terms that appeared in less than 10 documents.
## Currently working on combination 2 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,171 sparse dfm.
```

```
## Removing 1093 of 1171 total terms that appeared in less than 10 documents.
## Currently working on combination 3 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,155 sparse dfm.
```

```
## Removing 1088 of 1155 total terms that appeared in less than 10 documents.
## Currently working on combination 4 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,172 sparse dfm.
```

```
## Removing 1094 of 1172 total terms that appeared in less than 10 documents.
## Currently working on combination 5 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,158 sparse dfm.
```

```
## Removing 1091 of 1158 total terms that appeared in less than 10 documents.
## Currently working on combination 6 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,175 sparse dfm.
```

```
## Removing 1097 of 1175 total terms that appeared in less than 10 documents.
## Currently working on combination 7 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,159 sparse dfm.
```

```
## Removing 1092 of 1159 total terms that appeared in less than 10 documents.
## Currently working on combination 8 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,176 sparse dfm.
```

```
## Removing 1098 of 1176 total terms that appeared in less than 10 documents.
## Currently working on combination 9 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,404 sparse dfm.
```

```
## Removing 1350 of 1404 total terms that appeared in less than 10 documents.
## Currently working on combination 10 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,421 sparse dfm.
```

```
## Removing 1356 of 1421 total terms that appeared in less than 10 documents.
## Currently working on combination 11 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,405 sparse dfm.
```

```
## Removing 1351 of 1405 total terms that appeared in less than 10 documents.
## Currently working on combination 12 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,422 sparse dfm.
```

```
## Removing 1357 of 1422 total terms that appeared in less than 10 documents.
## Currently working on combination 13 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,404 sparse dfm.
```

```
## Removing 1350 of 1404 total terms that appeared in less than 10 documents.
## Currently working on combination 14 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,421 sparse dfm.
```

```
## Removing 1356 of 1421 total terms that appeared in less than 10 documents.
## Currently working on combination 15 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,405 sparse dfm.
```

```
## Removing 1351 of 1405 total terms that appeared in less than 10 documents.
## Currently working on combination 16 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,422 sparse dfm.
```

```
## Removing 1357 of 1422 total terms that appeared in less than 10 documents.
## Currently working on combination 17 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,264 sparse dfm.
```

```
## Removing 1124 of 1264 total terms that appeared in less than 10 documents.
## Currently working on combination 18 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,281 sparse dfm.
```

```
## Removing 1130 of 1281 total terms that appeared in less than 10 documents.
## Currently working on combination 19 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,265 sparse dfm.
```

```
## Removing 1125 of 1265 total terms that appeared in less than 10 documents.
## Currently working on combination 20 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,282 sparse dfm.
```

```
## Removing 1131 of 1282 total terms that appeared in less than 10 documents.
## Currently working on combination 21 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,268 sparse dfm.
```

```
## Removing 1128 of 1268 total terms that appeared in less than 10 documents.
## Currently working on combination 22 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,285 sparse dfm.
```

```
## Removing 1134 of 1285 total terms that appeared in less than 10 documents.
## Currently working on combination 23 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,269 sparse dfm.
```

```
## Removing 1129 of 1269 total terms that appeared in less than 10 documents.
## Currently working on combination 24 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,286 sparse dfm.
```

```
## Removing 1135 of 1286 total terms that appeared in less than 10 documents.
## Currently working on combination 25 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,519 sparse dfm.
```

```
## Removing 1392 of 1519 total terms that appeared in less than 10 documents.
## Currently working on combination 26 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,536 sparse dfm.
```

```
## Removing 1398 of 1536 total terms that appeared in less than 10 documents.
## Currently working on combination 27 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,520 sparse dfm.
```

```
## Removing 1393 of 1520 total terms that appeared in less than 10 documents.
## Currently working on combination 28 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,537 sparse dfm.
```

```
## Removing 1399 of 1537 total terms that appeared in less than 10 documents.
## Currently working on combination 29 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,519 sparse dfm.
```

```
## Removing 1392 of 1519 total terms that appeared in less than 10 documents.
## Currently working on combination 30 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,536 sparse dfm.
```

```
## Removing 1398 of 1536 total terms that appeared in less than 10 documents.
## Currently working on combination 31 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,520 sparse dfm.
```

```
## Removing 1393 of 1520 total terms that appeared in less than 10 documents.
## Currently working on combination 32 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,537 sparse dfm.
```

```
## Removing 1399 of 1537 total terms that appeared in less than 10 documents.
## Currently working on combination 33 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,154 sparse dfm.
```

```
## Currently working on combination 34 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,171 sparse dfm.
```

```
## Currently working on combination 35 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,155 sparse dfm.
```

```
## Currently working on combination 36 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,172 sparse dfm.
```

```
## Currently working on combination 37 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,158 sparse dfm.
```

```
## Currently working on combination 38 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,175 sparse dfm.
```

```
## Currently working on combination 39 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,159 sparse dfm.
```

```
## Currently working on combination 40 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,176 sparse dfm.
```

```
## Currently working on combination 41 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,404 sparse dfm.
```

```
## Currently working on combination 42 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,421 sparse dfm.
```

```
## Currently working on combination 43 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,405 sparse dfm.
```

```
## Currently working on combination 44 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,422 sparse dfm.
```

```
## Currently working on combination 45 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,404 sparse dfm.
```

```
## Currently working on combination 46 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,421 sparse dfm.
```

```
## Currently working on combination 47 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,405 sparse dfm.
```

```
## Currently working on combination 48 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,422 sparse dfm.
```

```
## Currently working on combination 49 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,264 sparse dfm.
```

```
## Currently working on combination 50 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,281 sparse dfm.
```

```
## Currently working on combination 51 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,265 sparse dfm.
```

```
## Currently working on combination 52 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,282 sparse dfm.
```

```
## Currently working on combination 53 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,268 sparse dfm.
```

```
## Currently working on combination 54 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,285 sparse dfm.
```

```
## Currently working on combination 55 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,269 sparse dfm.
```

```
## Currently working on combination 56 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,286 sparse dfm.
```

```
## Currently working on combination 57 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,519 sparse dfm.
```

```
## Currently working on combination 58 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,536 sparse dfm.
```

```
## Currently working on combination 59 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,520 sparse dfm.
```

```
## Currently working on combination 60 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0 seconds.
```

```
## Finished constructing a 1,000 x 1,537 sparse dfm.
```

```
## Currently working on combination 61 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,519 sparse dfm.
```

```
## Currently working on combination 62 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.02 seconds.
```

```
## Finished constructing a 1,000 x 1,536 sparse dfm.
```

```
## Currently working on combination 63 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.03 seconds.
```

```
## Finished constructing a 1,000 x 1,520 sparse dfm.
```

```
## Currently working on combination 64 of 64
```

```
## Creating a dfm from a tokens object...
```

```
##  ...lowercasing
```

```
##  ...complete, elapsed time: 0.01 seconds.
```

```
## Finished constructing a 1,000 x 1,537 sparse dfm.
```

``` r
# Run preText to evaluate preprocessing choices
pt <- preText(fp, dataset_name = "books_subset")
```

```
## Generating document distances...
## Currently working on dfm 1 of 64 
## Complete in: 1.11 seconds...
## Currently working on dfm 2 of 64 
## Complete in: 1.11 seconds...
## Currently working on dfm 3 of 64 
## Complete in: 1.27 seconds...
## Currently working on dfm 4 of 64 
## Complete in: 1.29 seconds...
## Currently working on dfm 5 of 64 
## Complete in: 1.06 seconds...
## Currently working on dfm 6 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 7 of 64 
## Complete in: 1.24 seconds...
## Currently working on dfm 8 of 64 
## Complete in: 1.18 seconds...
## Currently working on dfm 9 of 64 
## Complete in: 2.77 seconds...
## Currently working on dfm 10 of 64 
## Complete in: 1.27 seconds...
## Currently working on dfm 11 of 64 
## Complete in: 2.67 seconds...
## Currently working on dfm 12 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 13 of 64 
## Complete in: 2.68 seconds...
## Currently working on dfm 14 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 15 of 64 
## Complete in: 2.67 seconds...
## Currently working on dfm 16 of 64 
## Complete in: 1.25 seconds...
## Currently working on dfm 17 of 64 
## Complete in: 1.1 seconds...
## Currently working on dfm 18 of 64 
## Complete in: 1.25 seconds...
## Currently working on dfm 19 of 64 
## Complete in: 1.11 seconds...
## Currently working on dfm 20 of 64 
## Complete in: 1.07 seconds...
## Currently working on dfm 21 of 64 
## Complete in: 1.15 seconds...
## Currently working on dfm 22 of 64 
## Complete in: 1.18 seconds...
## Currently working on dfm 23 of 64 
## Complete in: 1.14 seconds...
## Currently working on dfm 24 of 64 
## Complete in: 1.13 seconds...
## Currently working on dfm 25 of 64 
## Complete in: 1.33 seconds...
## Currently working on dfm 26 of 64 
## Complete in: 1.06 seconds...
## Currently working on dfm 27 of 64 
## Complete in: 1.11 seconds...
## Currently working on dfm 28 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 29 of 64 
## Complete in: 1.12 seconds...
## Currently working on dfm 30 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 31 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 32 of 64 
## Complete in: 1.09 seconds...
## Currently working on dfm 33 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 34 of 64 
## Complete in: 1.11 seconds...
## Currently working on dfm 35 of 64 
## Complete in: 1.23 seconds...
## Currently working on dfm 36 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 37 of 64 
## Complete in: 1.1 seconds...
## Currently working on dfm 38 of 64 
## Complete in: 1.25 seconds...
## Currently working on dfm 39 of 64 
## Complete in: 1.07 seconds...
## Currently working on dfm 40 of 64 
## Complete in: 1.1 seconds...
## Currently working on dfm 41 of 64 
## Complete in: 1.09 seconds...
## Currently working on dfm 42 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 43 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 44 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 45 of 64 
## Complete in: 1.09 seconds...
## Currently working on dfm 46 of 64 
## Complete in: 1.09 seconds...
## Currently working on dfm 47 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 48 of 64 
## Complete in: 1.09 seconds...
## Currently working on dfm 49 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 50 of 64 
## Complete in: 1.11 seconds...
## Currently working on dfm 51 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 52 of 64 
## Complete in: 1.13 seconds...
## Currently working on dfm 53 of 64 
## Complete in: 1.12 seconds...
## Currently working on dfm 54 of 64 
## Complete in: 1.19 seconds...
## Currently working on dfm 55 of 64 
## Complete in: 1.26 seconds...
## Currently working on dfm 56 of 64 
## Complete in: 1.1 seconds...
## Currently working on dfm 57 of 64 
## Complete in: 1.11 seconds...
## Currently working on dfm 58 of 64 
## Complete in: 1.09 seconds...
## Currently working on dfm 59 of 64 
## Complete in: 1.09 seconds...
## Currently working on dfm 60 of 64 
## Complete in: 1.1 seconds...
## Currently working on dfm 61 of 64 
## Complete in: 1.08 seconds...
## Currently working on dfm 62 of 64 
## Complete in: 1.11 seconds...
## Currently working on dfm 63 of 64 
## Complete in: 1.07 seconds...
## Currently working on dfm 64 of 64 
## Complete in: 1.13 seconds...
## Generating preText Scores...
## Currently working on DFM: 1 of 64 
## 
## Complete in: 4.5 seconds...
## Currently working on DFM: 2 of 64 
## 
## Complete in: 4.48 seconds...
## Currently working on DFM: 3 of 64 
## 
## Complete in: 4.63 seconds...
## Currently working on DFM: 4 of 64 
## 
## Complete in: 5.45 seconds...
## Currently working on DFM: 5 of 64 
## 
## Complete in: 5.34 seconds...
## Currently working on DFM: 6 of 64 
## 
## Complete in: 4.58 seconds...
## Currently working on DFM: 7 of 64 
## 
## Complete in: 4.53 seconds...
## Currently working on DFM: 8 of 64 
## 
## Complete in: 5.21 seconds...
## Currently working on DFM: 9 of 64 
## 
## Complete in: 5.15 seconds...
## Currently working on DFM: 10 of 64 
## 
## Complete in: 4.74 seconds...
## Currently working on DFM: 11 of 64 
## 
## Complete in: 4.73 seconds...
## Currently working on DFM: 12 of 64 
## 
## Complete in: 4.47 seconds...
## Currently working on DFM: 13 of 64 
## 
## Complete in: 4.47 seconds...
## Currently working on DFM: 14 of 64 
## 
## Complete in: 4.44 seconds...
## Currently working on DFM: 15 of 64 
## 
## Complete in: 4.61 seconds...
## Currently working on DFM: 16 of 64 
## 
## Complete in: 4.39 seconds...
## Currently working on DFM: 17 of 64 
## 
## Complete in: 4.37 seconds...
## Currently working on DFM: 18 of 64 
## 
## Complete in: 4.39 seconds...
## Currently working on DFM: 19 of 64 
## 
## Complete in: 4.38 seconds...
## Currently working on DFM: 20 of 64 
## 
## Complete in: 4.39 seconds...
## Currently working on DFM: 21 of 64 
## 
## Complete in: 4.61 seconds...
## Currently working on DFM: 22 of 64 
## 
## Complete in: 4.44 seconds...
## Currently working on DFM: 23 of 64 
## 
## Complete in: 4.4 seconds...
## Currently working on DFM: 24 of 64 
## 
## Complete in: 4.41 seconds...
## Currently working on DFM: 25 of 64 
## 
## Complete in: 4.45 seconds...
## Currently working on DFM: 26 of 64 
## 
## Complete in: 4.64 seconds...
## Currently working on DFM: 27 of 64 
## 
## Complete in: 4.92 seconds...
## Currently working on DFM: 28 of 64 
## 
## Complete in: 4.53 seconds...
## Currently working on DFM: 29 of 64 
## 
## Complete in: 4.6 seconds...
## Currently working on DFM: 30 of 64 
## 
## Complete in: 4.8 seconds...
## Currently working on DFM: 31 of 64 
## 
## Complete in: 4.64 seconds...
## Currently working on DFM: 32 of 64 
## 
## Complete in: 4.46 seconds...
## Currently working on DFM: 33 of 64 
## 
## Complete in: 4.61 seconds...
## Currently working on DFM: 34 of 64 
## 
## Complete in: 4.49 seconds...
## Currently working on DFM: 35 of 64 
## 
## Complete in: 4.42 seconds...
## Currently working on DFM: 36 of 64 
## 
## Complete in: 4.41 seconds...
## Currently working on DFM: 37 of 64 
## 
## Complete in: 4.43 seconds...
## Currently working on DFM: 38 of 64 
## 
## Complete in: 4.44 seconds...
## Currently working on DFM: 39 of 64 
## 
## Complete in: 4.6 seconds...
## Currently working on DFM: 40 of 64 
## 
## Complete in: 4.46 seconds...
## Currently working on DFM: 41 of 64 
## 
## Complete in: 4.44 seconds...
## Currently working on DFM: 42 of 64 
## 
## Complete in: 5.02 seconds...
## Currently working on DFM: 43 of 64 
## 
## Complete in: 5.31 seconds...
## Currently working on DFM: 44 of 64 
## 
## Complete in: 5.09 seconds...
## Currently working on DFM: 45 of 64 
## 
## Complete in: 5.05 seconds...
## Currently working on DFM: 46 of 64 
## 
## Complete in: 4.89 seconds...
## Currently working on DFM: 47 of 64 
## 
## Complete in: 4.77 seconds...
## Currently working on DFM: 48 of 64 
## 
## Complete in: 4.72 seconds...
## Currently working on DFM: 49 of 64 
## 
## Complete in: 4.6 seconds...
## Currently working on DFM: 50 of 64 
## 
## Complete in: 4.41 seconds...
## Currently working on DFM: 51 of 64 
## 
## Complete in: 4.61 seconds...
## Currently working on DFM: 52 of 64 
## 
## Complete in: 4.44 seconds...
## Currently working on DFM: 53 of 64 
## 
## Complete in: 4.53 seconds...
## Currently working on DFM: 54 of 64 
## 
## Complete in: 4.41 seconds...
## Currently working on DFM: 55 of 64 
## 
## Complete in: 4.37 seconds...
## Currently working on DFM: 56 of 64 
## 
## Complete in: 4.36 seconds...
## Currently working on DFM: 57 of 64 
## 
## Complete in: 4.67 seconds...
## Currently working on DFM: 58 of 64 
## 
## Complete in: 4.38 seconds...
## Currently working on DFM: 59 of 64 
## 
## Complete in: 4.45 seconds...
## Currently working on DFM: 60 of 64 
## 
## Complete in: 4.39 seconds...
## Currently working on DFM: 61 of 64 
## 
## Complete in: 4.42 seconds...
## Currently working on DFM: 62 of 64 
## 
## Complete in: 4.49 seconds...
## Currently working on DFM: 63 of 64 
## 
## Complete in: 4.72 seconds...
## Currently working on DFM: 64 of 64 
## Complete in: 0 seconds...
## Generating regression results..
## The R^2 for this model is: 0.3174779 
## Regression results (negative coefficients imply less risk):
##                  Variable Coefficient    SE
## 1               Intercept       0.055 0.013
## 2      Remove Punctuation       0.015 0.009
## 3          Remove Numbers      -0.003 0.009
## 4               Lowercase       0.003 0.009
## 5                Stemming       0.008 0.009
## 6        Remove Stopwords       0.030 0.009
## 7 Remove Infrequent Terms       0.035 0.009
## Complete in: 371.24 seconds...
```

``` r
# Visualize preprocessing robustness
preText_score_plot(pt)
```

```
## Warning in ggplot2::geom_point(ggplot2::aes(x = Variable, y = Coefficient), :
## Ignoring unknown parameters: `linewidth`
```

![](unsupervised-topicmodels_files/figure-latex/exercise3-1.pdf)<!-- --> 
The preText score plot reveals the relative impact of different pre-processing combinations on the consistency of text representations. Combinations with scores closer to zero are considered low-risk, as they produce results broadly consistent with the majority of other approaches. In contrast, combinations positioned towards the right of the plot, particularly those incorporating both stopword removal and infrequent term removal simultaneously, yield substantially more atypical results and therefore carry a higher degree of analytical risk.

