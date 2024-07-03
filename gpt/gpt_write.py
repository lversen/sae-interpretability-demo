from transformers import pipeline
from movie_dataset_new import movie_dict

generator = pipeline('text-generation', model='gpt2')

summaries = {}
length = 1000
for movie in movie_dict.values():
    try:
        movie = movie[0]
        summary = open("gpt_summaries\summarygpt_" + movie + ".txt", "w")
        g = generator("After this input give a synopsis of this movie: " +
                      movie, max_length=length, truncation=True)
        g = g[0]["generated_text"].replace(
            "After this input give a summary of this movie: " + movie, "")
        summary.write(g)
        summary.close()
    except UnicodeEncodeError:
        print(movie + " was skipped")
