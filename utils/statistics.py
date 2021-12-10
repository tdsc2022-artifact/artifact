def print_c2s_distribution(cweid):
    ct0, ct1 = 0, 0
    with open(f"data/code2seq/{cweid}/all.c2s", "r") as f:
        for line in f.readlines():
            label, *ctxs = line.split()
            if label == "0":
                ct0 += 1
            if label == "1":
                ct1 += 1
    print(f"0: {ct0}")
    print(f"1: {ct1}")

def print_token_distribution(cweid):
    ct0, ct1 = 0, 0
    with open(f"data/token/{cweid}/{cweid}.txt", "r") as f:
        for line in f.readlines():
            label, *ctxs = line.split()
            if label == "0":
                ct0 += 1
            if label == "1":
                ct1 += 1
    print(f"0: {ct0}")
    print(f"1: {ct1}")