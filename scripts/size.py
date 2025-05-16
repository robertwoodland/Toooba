import sys

def perceptron_calculation(local, gLength, size, weight):
    lHist = size * local
    gHist = gLength
    weights = size * (local + 1 + gLength) * weight # local + global + bias
        
    totalBits = lHist + gHist + weights
    
    return (totalBits // 8) # Bytes

def bht_calculation(entries, width):
    return (entries * width) // 8

def tournament_calculation(local, gLength, pcSize):
    localTab = (2**pcSize) * local # Local history table
    localBht = (2**local) * 3
    BHTs = 2**(gLength+2) # Global BHT and choice BHT
    
    total = gLength + localTab + localBht + BHTs

    return (total // 8)

def main():
    if len(sys.argv) < 3:
        print("Usage: python size.py <predictor> <value>")
        sys.exit(1)

    predictor = sys.argv[1]
    values = sys.argv[2:] # Probs dangerous but oh well

    if predictor == "perceptron":
        try:
            local = int(values[0])
            gLength = int(values[1])
            size = int(values[2])
            if (len(values) > 3):
                width = int(values[3])
            else:
                width = 8
            result = perceptron_calculation(local, gLength, size, width)
        except ValueError:
            print("Error: Value must be three integers for perceptron, of the form <local> <global> <size>")
            sys.exit(1)
            
    elif predictor == "bht":
        try:
            entries = int(values[0])
            if (len(values) > 1):
                width = int(values[1])
            else:
                width = 2
                
            result = bht_calculation(entries, width)
        except ValueError:
            print("Error: Value must be two integers for bht, of the form <entries> <width>")
            sys.exit(1)
            
    elif predictor == "tour":
        try:
            local = int(values[0])
            gLength = int(values[1])
            pcSize = int(values[2])
            result = tournament_calculation(local, gLength, pcSize)
        except ValueError:
            print("Error: Value must be three integers for tournament, of the form <local> <global> <pcSize>")
            sys.exit(1)
    else:
        print("Error: Predictor must be one of perceptron, bht, tournament.")
        sys.exit(1)

    print(result)

if __name__ == "__main__":
    main()
