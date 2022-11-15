from nltk.tokenize import sent_tokenize



def pre_rouge_processing(summary):

    summary = summary.replace("<n>", " ")    
    summary = "\n".join(sent_tokenize(summary))
    
    return summary