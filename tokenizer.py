import sys
import re

def textfromfile(inputfile):
    with open(inputfile, 'r') as f:
        text = f.read()
    return text

def splitwords(longtext):
    #vector = longtext.split()
    vector = re.split(r'([,.:;?_!"()\']|--|\s)', longtext)
    vector = [item.strip() for item in vector if item.strip()]
    return(vector)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inputfile = sys.argv[1]
        longtext = textfromfile(inputfile)
        print("Input text has {chars} characters".format(chars=len(longtext)))

        tokenized = splitwords(longtext)
        print("Tokenized text has {tokens} tokens".format(tokens=len(tokenized)))
        print(tokenized[:100])
    else:
        print("Please provide an input file")