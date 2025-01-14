import sys
import re
import tiktoken

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

        tokenizer = tiktoken.get_encoding("gpt2")
        ids = tokenizer.encode(longtext, allowed_special={"<|endoftext|>"})

        # text1 = "Hello, do you like tea?"
        # text2 = "In the sunlit terraces of the someunknownPlace."
        # text = " <|endoftext|> ".join((text1, text2))
        text = "Akwirw ier"
        
        integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        #print(integers)

        tokens = [tokenizer.decode([i]) for i in integers]
        #print(tokens)

        mapping = [(i,s) for i,s in zip(integers, tokens)]
        print(mapping)

        strings = tokenizer.decode(integers)
        print(strings)

    else:
        print("Please provide an input file")