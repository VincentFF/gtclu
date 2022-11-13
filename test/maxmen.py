with open("./memeory.txt") as fl:
    res = 0
    line = fl.readline().strip()
    while line:
        var = int(line[:-1])
        res = max(var, res)
        print(res)
        line = fl.readline().strip()
