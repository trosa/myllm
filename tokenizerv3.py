import sys
import tiktoken

def textfromfile(inputfile):
    with open(inputfile, 'r') as f:
        text = f.read()
    return text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inputfile = sys.argv[1]
        longtext = textfromfile(inputfile)
        
        tokenizer = tiktoken.get_encoding("gpt2")
        encoded_text = tokenizer.encode(longtext)
        enc_sample = encoded_text[50:]
        
        context_size = 4
        x = enc_sample[:context_size]
        y = enc_sample[1:context_size+1]
        print(x)
        print(y)

        for i in range(1, context_size+1):
            context = enc_sample[:i]
            desired = enc_sample[i]
            print(tokenizer.decode(context), "----->", tokenizer.decode([desired]))

    else:
        print("Please provide an input file")