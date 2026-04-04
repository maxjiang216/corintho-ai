def main():
    res = {}

    for i in range(200):
        with open(f"round_{i}/results.txt", "r") as f:
            results = f.read().split("\n")
        results = [r.strip().split() for r in results]
        results = [r for r in results if len(r) == 3]
        for r in results:
            a, b, c = float(r[0]), float(r[1]), float(r[2])
            if (a, b) not in res:
                res[(a, b)] = [0, 0, 0]
            if c == 1:
                res[(a, b)][0] += 1
            elif c == 0.5:
                res[(a, b)][1] += 1
            else:
                res[(a, b)][2] += 1

    with open("results.txt", "w") as f:
        for k, v in res.items():
            f.write(f"{int(k[0])}\t{int(k[1])}\t{v[0]}\t{v[1]}\t{v[2]}\n")

    lst = []
    for k, v in res.items():
        lst.append((k, sum(v)))

    lst.sort(key=lambda x: x[1])

    lst2 = []
    for l in lst:
        if l[1] <= 80:
            lst2.append((abs(l[0][0] - l[0][1]), l[0][0], l[0][1]))

    lst2.sort()

    print(lst2)


if __name__ == "__main__":
    main()
