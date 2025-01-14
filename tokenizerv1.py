import sys
import re

def textfromfile(inputfile):
    with open(inputfile, 'r') as f:
        text = f.read()
    return text

def splitwords(longtext):
    vector = re.split(r'([,.:;?_!"()\']|--|\s)', longtext)
    vector = [item.strip() for item in vector if item.strip()]
    return vector

def create_vocabulary(tokenized):
    all_tokens = sorted(list(set(tokenized)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer,token in enumerate(all_tokens)}
    return vocab

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, longtext):
        vector = re.split(r'([,.:;?_!"()\']|--|\s)', longtext)
        vector = [item.strip() for item in vector if item.strip()]
        vector = [item if item in self.str_to_int else "<|unk|>" for item in vector]
        ids = [self.str_to_int[s] for s in vector]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inputfile = sys.argv[1]
        longtext = textfromfile(inputfile)
        print("Input text has {chars} characters".format(chars=len(longtext)))

        tokenized = splitwords(longtext)
        print("Tokenized text has {tokens} tokens".format(tokens=len(tokenized)))
        print(tokenized[:100])

        vocab = create_vocabulary(tokenized)
        
        for index, token in enumerate(list(vocab.items())[-5:]):
            print(token)

        print("Vocabulary size is", len(vocab))

        tokenizer = SimpleTokenizerV2(vocab)
        ids = tokenizer.encode(longtext)

        text1 = "Hello, do you like tea?"
        text2 = "In the sunlit terraces of the palace."
        text = " <|endoftext|> ".join((text1, text2))
        
        print(tokenizer.encode(text))
        print(tokenizer.decode(tokenizer.encode(text)))

    else:
        print("Please provide an input file")