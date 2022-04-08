from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import word_tokenize
import re
import nltk
nltk.download('punkt')
text = "William Bronk is best known for his austere view of the world as well as writing style. His language—subtle, balanced in tone and diction, essential—is possibly the most distilled in all of twentieth-century American poetry. In addition, Bronk is always explicit visually and resonant musically. His work keeps alive a New England poetic tradition, evoking nature and the seasons, winter most of all, and delving into the nature of reality or truth. These concerns were firmly established early in twentieth-century American poetry by the New England poets Robert FROST and Wallace STEVENS, then later by, along with Bronk, Robert CREELEY and George OPPEN, and in the nineteenth century by Henry David Thoreau (an especially strong influence on Bronk), Ralph Waldo Emerson, and Emily Dickinson"
def summarise(text):
    model = AutoModelForSeq2SeqLM.from_pretrained("Linguist/t5-small-Linguists_summariser")
    tokenizer = AutoTokenizer.from_pretrained("Linguist/t5-small-Linguists_summariser")
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=150, min_length=20, length_penalty=2.0, num_beams=4)#, early_stopping=True)
    op_text = tokenizer.decode(outputs[0])
    pattern = "^<pad>.*</s>$"
    result = re.sub(r'(^<pad>)|(</s>$)', '', op_text)
    return result #op_text
def combine(text):
    lst = word_tokenize(text)
    l = len(lst)
    mega = []
    chunks = (l//200)+1
    temp = ""
    t = []
    for i in range(chunks):
        t = lst[i:i+200]
        temp = " ".join(t)
        mega.append(summarise(temp))
    print(mega)
    m_txt = " ".join(mega)
    return m_txt
print(combine(text))

